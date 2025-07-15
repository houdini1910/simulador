import streamlit as st
import numpy as np
from math import factorial
from fpdf import FPDF

st.set_page_config(page_title="Simulador de Colas y Monte Carlo", layout="centered")

# --- Diccionario de explicaciones ---
EXPLICACIONES = {
    "Modelo": "Tipo de sistema de colas utilizado",
    "lambda": "λ (lambda) - Tasa de llegada (clientes por unidad de tiempo)",
    "mu": "μ (mu) - Tasa de servicio (clientes atendidos por servidor por unidad de tiempo)",
    "c": "c - Número de servidores",
    "K": "K - Capacidad máxima total del sistema (incluye en servicio y en cola)",
    "rho": "ρ (rho) - Utilización del sistema (porcentaje de tiempo ocupado)",
    "P0": "P₀ - Probabilidad de que no haya clientes en el sistema",
    "Lq": "Lq - Número promedio de clientes en la cola",
    "Ls": "Ls - Número promedio de clientes en el sistema (cola + servicio)",
    "Wq": "Wq - Tiempo promedio en cola (espera)",
    "Ws": "Ws - Tiempo promedio en el sistema (espera + servicio)",
    "lambda_eff": "λₑₓₓ - Tasa efectiva de llegada (clientes que realmente entran al sistema)",
    "Distribucion": "Distribución de probabilidad P(n) y acumulada para cada número de clientes en el sistema"
}

# ------ FUNCIONES DE CÁLCULO ------
def calcular_mm1(lmbda, mu):
    if lmbda >= mu:
        raise ValueError("Para el modelo M/M/1, λ debe ser menor que μ para que el sistema sea estable.")
    rho = lmbda / mu
    P0 = 1 - rho
    Lq = rho**2 / (1 - rho)
    Ls = Lq + rho
    Wq = Lq / lmbda
    Ws = Ls / lmbda
    return {"Modelo": "M/M/1", "lambda": lmbda, "mu": mu, "rho": rho, "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws}

def calcular_mmc(lmbda, mu, c):
    if c < 1 or lmbda < 0 or mu <= 0:
        raise ValueError("Revisa los datos. Debe ser c ≥ 1, λ ≥ 0 y μ > 0.")
    rho = lmbda / (c * mu)
    if rho >= 1:
        raise ValueError("Para el modelo M/M/c, λ/(cμ) debe ser menor que 1 para que el sistema sea estable.")
    suma = sum((lmbda/mu)**n / factorial(n) for n in range(c))
    complemento = (lmbda/mu)**c / (factorial(c) * (1 - rho))
    P0 = 1 / (suma + complemento)
    Lq = (P0 * (lmbda/mu)**c * rho) / (factorial(c) * (1 - rho)**2)
    Ls = Lq + c * rho
    Wq = Lq / lmbda
    Ws = Wq + 1/mu
    return {"Modelo": f"M/M/{c}", "lambda": lmbda, "mu": mu, "c": c, "rho": rho, "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws}

def calcular_mmck(lmbda, mu, c, K):
    if c < 1 or K < c or lmbda < 0 or mu <= 0:
        raise ValueError("Revisa los datos. K ≥ c ≥ 1, λ ≥ 0 y μ > 0.")
    rho_s = lmbda / mu
    terms = [(rho_s**n) / factorial(n) for n in range(c)]
    factor = rho_s**c / factorial(c)
    cola_terms = [(rho_s / c)**m for m in range(1, K - c + 1)]
    P0 = 1 / (sum(terms) + factor * sum(cola_terms))
    P = [(rho_s**n) / factorial(n) * P0 if n < c
         else (rho_s**c) / factorial(c) * (rho_s / c)**(n - c) * P0
         for n in range(K + 1)]
    Ls = sum(n * pn for n, pn in enumerate(P))
    Lq = sum((n - c) * pn for n, pn in enumerate(P) if n > c)
    lambda_eff = lmbda * (1 - P[-1])
    if lambda_eff == 0:
        raise ValueError("λ_eff=0. Verifica que λ y K tengan sentido para tu sistema.")
    Wq = Lq / lambda_eff
    Ws = Wq + 1/mu
    cumul = []
    total = 0
    for pn in P:
        total += pn
        cumul.append(total)
    return {"Modelo": f"M/M/{c}/{K}", "lambda": lmbda, "mu": mu, "c": c, "K": K, "rho": rho_s / c,
            "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws,
            "lambda_eff": lambda_eff, "Distribucion": list(zip(P, cumul))}

# ------ PDF GENERATION ------
def strip_unicode(text):
    return str(text).replace("—", "-").replace("–", "-").replace("λ", "lambda").replace("μ", "mu")

def generar_pdf(result_dict, filename="reporte_simulacion.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Reporte de Simulacion de Colas", ln=1, align='C')
    pdf.ln(8)
    for k, v in result_dict.items():
        if k != "Distribucion":
            if isinstance(v, float):
                v_str = f"{v:.4f}"
            else:
                v_str = str(v)
            nombre = strip_unicode(EXPLICACIONES.get(k, k))
            pdf.cell(0, 8, f"{nombre}: {strip_unicode(v_str)}", ln=1)
    pdf.ln(5)
    if "Distribucion" in result_dict:
        dist = result_dict["Distribucion"]
        pdf.set_font("Arial", size=11, style='B')
        pdf.cell(0, 8, "Distribucion P(n) y acumulada:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.cell(20, 8, "n", border=1)
        pdf.cell(40, 8, "P(n)", border=1)
        pdf.cell(40, 8, "Acumulada", border=1)
        pdf.ln()
        for i, (p, ac) in enumerate(dist):
            pdf.cell(20, 8, f"{i}", border=1)
            pdf.cell(40, 8, f"{p:.4f}", border=1)
            pdf.cell(40, 8, f"{ac:.4f}", border=1)
            pdf.ln()
    b = pdf.output(dest='S').encode('latin1')
    return b

# --- Interfaz principal ---
st.markdown("""
    <style>
    [data-testid="stHeadingLink"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Simulador de Colas y Monte Carlo")

tabs = st.tabs([
    "Modelos de Colas",
    "Simulación Monte Carlo",
    "Asistente",
    "Ayuda y Conceptos"
])

# -------- PESTAÑA 1: MODELOS CLÁSICOS
with tabs[0]:
    st.header("Simulación de Modelos de Colas")
    lmbda = st.number_input("λ (Tasa de llegada)", min_value=0.01, value=1.5, format="%.2f", help="Clientes que llegan por unidad de tiempo")
    mu = st.number_input("μ (Tasa de servicio)", min_value=0.01, value=2.5, format="%.2f", help="Clientes atendidos por servidor por unidad de tiempo")
    c = st.number_input("Cantidad de servidores (c)", min_value=1, step=1, value=1, help="Número de servidores")

    limitar = st.checkbox("Limitar capacidad de la cola (M/M/c/K)")
    if limitar:
        K = st.number_input("Capacidad total K", min_value=int(c), step=1, value=int(c)+3, help="Capacidad máxima del sistema")
    else:
        K = None

    resultado = None
    if st.button("Calcular"):
        try:
            if lmbda <= 0 or mu <= 0:
                st.error("λ y μ deben ser mayores que 0.")
            elif limitar and (K is None or K < c):
                st.error("K debe ser mayor o igual que c.")
            else:
                # Validaciones según el modelo
                if limitar:
                    resultado = calcular_mmck(lmbda, mu, int(c), int(K))
                else:
                    if int(c) == 1:
                        resultado = calcular_mm1(lmbda, mu)
                    else:
                        resultado = calcular_mmc(lmbda, mu, int(c))
                st.success("¡Cálculo realizado con éxito!")
                for k, v in resultado.items():
                    nombre = EXPLICACIONES.get(k, k)
                    if k == "Distribucion":
                        st.markdown(f"**{nombre}:**")
                        st.table(
                            [{"n": i, "P(n)": round(p,4), "Acumulada": round(ac,4)} for i, (p, ac) in enumerate(v)]
                        )
                    else:
                        valor = f"{v:.4f}" if isinstance(v, float) else v
                        st.markdown(f"**{nombre}:** {valor}")
                if resultado:
                    pdf_bytes = generar_pdf(resultado)
                    st.download_button(
                        label="Descargar reporte en PDF",
                        data=pdf_bytes,
                        file_name="reporte_simulacion.pdf",
                        mime="application/pdf"
                    )
        except Exception as ex:
            st.error(f"Error: {ex}")

# -------- PESTAÑA 2: MONTE CARLO
with tabs[1]:
    st.header("Simulación de Monte Carlo (Poisson / Exponencial)")
    dist = st.radio("Selecciona la distribución", ["Poisson", "Exponencial"])
    lmbda_mc = st.number_input("λ (tasa promedio)", min_value=0.01, value=1.0, format="%.2f", help="Promedio de ocurrencias")
    n_vars = st.number_input("Cantidad de variables a simular", min_value=1, value=5, help="Columnas de la simulación")
    n_obs = st.number_input("Cantidad de observaciones", min_value=1, value=3, help="Filas de la simulación")
    
    if st.button("Simular Monte Carlo"):
        try:
            if lmbda_mc <= 0:
                st.error("λ debe ser mayor a 0.")
            else:
                if dist == "Poisson":
                    resultados = np.random.poisson(lmbda_mc, size=(n_obs, n_vars))
                else:
                    resultados = np.random.exponential(1/lmbda_mc, size=(n_obs, n_vars))

                st.write("**Primeras 5 simulaciones:**")
                st.write(resultados[:5])

                st.info(f"Media total: {np.mean(resultados):.4f}")
                st.info(f"Desviación estándar: {np.std(resultados):.4f}")

                encabezado = ','.join([f"Var{i+1}" for i in range(resultados.shape[1])])
                lineas = [encabezado]
                for fila in resultados:
                    lineas.append(','.join([str(x) for x in fila]))
                csv_data = '\n'.join(lineas)

                st.download_button(
                    label="Descargar resultados (CSV)",
                    data=csv_data,
                    file_name="resultados_montecarlo.csv",
                    mime="text/csv"
                )
        except Exception as ex:
            st.error(f"Error: {ex}")

# -------- PESTAÑA 3: ASISTENTE MEJORADO
with tabs[2]:
    st.markdown("<h2 style='color:#0d47a1;font-weight:bold'>Asistente Virtual 👩‍💻</h2>", unsafe_allow_html=True)
    st.markdown("> <span style='color:#1565c0;font-weight:bold'>¡Sigue los pasos para resolver tu problema de colas!</span>", unsafe_allow_html=True)

    modelos = {
        "M/M/1": {
            "desc": "Un solo servidor, cola ilimitada.",
            "ej": "🧑‍💼 <b>Ejemplo:</b> Un cajero atendiendo en un banco sin límite de espera."
        },
        "M/M/1/K": {
            "desc": "Un solo servidor, capacidad limitada.",
            "ej": "🪑 <b>Ejemplo:</b> Sala de espera con solo 5 asientos."
        },
        "M/M/c": {
            "desc": "Varios servidores, cola ilimitada.",
            "ej": "🏢 <b>Ejemplo:</b> 3 médicos atendiendo pacientes en una clínica."
        },
        "M/M/c/K": {
            "desc": "Varios servidores, capacidad limitada.",
            "ej": "📞 <b>Ejemplo:</b> 5 líneas en un call center con máximo 10 personas en total."
        }
    }

    if 'asist_paso' not in st.session_state:
        st.session_state.asist_paso = 1
    if 'asist_modelo' not in st.session_state:
        st.session_state.asist_modelo = None
    if 'asist_lambda' not in st.session_state:
        st.session_state.asist_lambda = None
    if 'asist_mu' not in st.session_state:
        st.session_state.asist_mu = None
    if 'asist_c' not in st.session_state:
        st.session_state.asist_c = None
    if 'asist_K' not in st.session_state:
        st.session_state.asist_K = None
    if 'asist_result' not in st.session_state:
        st.session_state.asist_result = None

    # Paso 1: Modelo
    if st.session_state.asist_paso == 1:
        st.markdown("<h4 style='color:#0277bd'>1️⃣ Selecciona el tipo de modelo</h4>", unsafe_allow_html=True)
        for nombre, data in modelos.items():
            if st.button(f"Elegir {nombre}", key=f"asist_elegir_{nombre}"):
                st.session_state.asist_modelo = nombre
                st.session_state.asist_paso = 2
                st.session_state.asist_lambda = None
                st.session_state.asist_mu = None
                st.session_state.asist_c = None
                st.session_state.asist_K = None
                st.session_state.asist_result = None
        for nombre, data in modelos.items():
            st.markdown(
                f"<div style='background-color:#e3f2fd; padding:10px; margin-bottom:8px; border-radius:8px;'>"
                f"<b style='color:#01579b;font-size:18px;'>{nombre}</b>: "
                f"<span style='color:#263238;font-size:16px;'>{data['desc']}</span><br>"
                f"<span style='color:#1565c0; font-size:15px;'>{data['ej']}</span></div>",
                unsafe_allow_html=True)
        st.markdown(
            "<div style='background-color:#1565c0;padding:10px;border-radius:8px;color:white;font-size:15px;'><b>💡 Elige el modelo que más se parece a tu situación real.</b></div>",
            unsafe_allow_html=True)

    # Paso 2: Lambda
    elif st.session_state.asist_paso == 2:
        st.markdown("<h4 style='color:#0277bd'>2️⃣ Ingresa la tasa de llegada λ (lambda)</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#f3e5f5;padding:10px;border-radius:8px;'>
        <b style='color:#6a1b9a'>¿Qué es lambda?</b>
        <span style='color:#222'> Es el número promedio de clientes que llegan por unidad de tiempo.<br>
        <b>Ejemplo:</b> Si cada 2 minutos llegan 4 personas, entonces lambda = 2 por minuto.<br>
        <b style='color:#00897b;'>TIP:</b> ¿Cuántos clientes llegan en 1 hora? Divide por 60 si quieres el valor por minuto.</span>
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("λ (tasa de llegada)", min_value=0.01, value=1.0, format="%.2f", key="asist_lambda_input")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_lambda_paso2"):
            if val <= 0:
                st.warning("λ debe ser mayor que 0.")
            else:
                st.session_state.asist_lambda = val
                st.session_state.asist_paso = 3
        if col2.button("⬅️ Volver", key="asist_volver_modelo_paso2"):
            st.session_state.asist_paso = 1

    # Paso 3: Mu
    elif st.session_state.asist_paso == 3:
        st.markdown("<h4 style='color:#0277bd'>3️⃣ Ingresa la tasa de servicio μ (mu)</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#fff9c4;padding:10px;border-radius:8px;'>
        <b style='color:#f57c00'>¿Qué es mu?</b>
        <span style='color:#333'> Es el número promedio de clientes que un servidor puede atender por unidad de tiempo.<br>
        <b>Ejemplo:</b> Si cada médico atiende 5 personas por hora, entonces mu = 5 por hora.<br>
        <b style='color:#00897b;'>TIP:</b> Si tienes varios servidores y todos atienden igual, pon la tasa individual aquí.</span>
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("μ (tasa de servicio)", min_value=0.01, value=2.0, format="%.2f", key="asist_mu_input")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_mu_paso3"):
            if val <= 0:
                st.warning("μ debe ser mayor que 0.")
            else:
                st.session_state.asist_mu = val
                if st.session_state.asist_modelo in ["M/M/c", "M/M/c/K"]:
                    st.session_state.asist_paso = 4
                elif st.session_state.asist_modelo == "M/M/1/K":
                    st.session_state.asist_paso = 5
                else:
                    st.session_state.asist_paso = 6
        if col2.button("⬅️ Volver", key="asist_volver_lambda_paso3"):
            st.session_state.asist_paso = 2

    # Paso 4: c (servidores)
    elif st.session_state.asist_paso == 4:
        st.markdown("<h4 style='color:#0277bd'>4️⃣ Ingresa la cantidad de servidores c</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#ffe0b2;padding:10px;border-radius:8px;'>
        <b style='color:#bf360c'>¿Qué es c?</b>
        <span style='color:#212121'> Es el número de servidores o puestos que atienden simultáneamente.<br>
        <b>Ejemplo:</b> 4 ventanillas en un banco, c = 4.<br>
        <b style='color:#00897b;'>TIP:</b> Si tienes un solo servidor, pon c=1 y usa el modelo M/M/1.</span>
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("Cantidad de servidores (c)", min_value=1, value=2, step=1, key="asist_c_input")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_c_paso4"):
            if val < 1:
                st.warning("Debe haber al menos 1 servidor.")
            else:
                st.session_state.asist_c = val
                if st.session_state.asist_modelo == "M/M/c/K":
                    st.session_state.asist_paso = 5
                else:
                    st.session_state.asist_paso = 6
        if col2.button("⬅️ Volver", key="asist_volver_mu_paso4"):
            st.session_state.asist_paso = 3

    # Paso 5: K (capacidad máxima)
    elif st.session_state.asist_paso == 5:
        st.markdown("<h4 style='color:#0277bd'>5️⃣ Ingresa la capacidad máxima del sistema K</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color:#e8f5e9;padding:10px;border-radius:8px;'>
        <b style='color:#388e3c'>¿Qué es K?</b>
        <span style='color:#232'> Es el máximo número de personas que pueden estar en el sistema (esperando + en servicio).<br>
        <b>Ejemplo:</b> 1 cajero y 5 sillas: K = 6.<br>
        <b style='color:#00897b;'>TIP:</b> Si no hay límite, usa los modelos sin K.</span>
        </div>
        """, unsafe_allow_html=True)
        min_c = int(st.session_state.asist_c) if st.session_state.asist_c else 1
        val = st.number_input("Capacidad total (K)", min_value=min_c, value=min_c + 3, step=1, key="asist_K_input")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_K_paso5"):
            if val < min_c:
                st.warning("K debe ser mayor o igual que c.")
            else:
                st.session_state.asist_K = val
                st.session_state.asist_paso = 6
        if col2.button("⬅️ Volver", key="asist_volver_cK_paso5"):
            if st.session_state.asist_modelo == "M/M/1/K":
                st.session_state.asist_paso = 3
            else:
                st.session_state.asist_paso = 4

    # Paso 6: Resultados
    elif st.session_state.asist_paso == 6:
        st.markdown("<h3 style='color:#388e3c'>Resultados y análisis 📊</h3>", unsafe_allow_html=True)
        modelo = st.session_state.asist_modelo
        try:
            lmbda = float(st.session_state.asist_lambda) if st.session_state.asist_lambda is not None else 1.0
            mu = float(st.session_state.asist_mu) if st.session_state.asist_mu is not None else 1.0
            c = int(st.session_state.asist_c) if st.session_state.asist_c else 1
            K = int(st.session_state.asist_K) if st.session_state.asist_K else None
            resultado = None
            # Validaciones lógicas
            if lmbda <= 0 or mu <= 0 or c < 1 or (K is not None and K < c):
                st.error("Revisa que todos los datos sean mayores que cero y K ≥ c.")
            else:
                if modelo == "M/M/1":
                    resultado = calcular_mm1(lmbda, mu)
                elif modelo == "M/M/c":
                    resultado = calcular_mmc(lmbda, mu, int(c))
                elif modelo == "M/M/1/K":
                    resultado = calcular_mmck(lmbda, mu, 1, int(K))
                elif modelo == "M/M/c/K":
                    resultado = calcular_mmck(lmbda, mu, int(c), int(K))
                if resultado:
                    st.success("¡Cálculo realizado con éxito! Mira el significado de cada resultado 👇")
                    for k, v in resultado.items():
                        nombre = EXPLICACIONES.get(k, k)
                        if k == "Distribucion":
                            st.markdown(f"**{nombre}:**")
                            st.table(
                                [{"n": i, "P(n)": round(p,4), "Acumulada": round(ac,4)} for i, (p, ac) in enumerate(v)]
                            )
                        else:
                            valor = f"{v:.4f}" if isinstance(v, float) else v
                            st.markdown(
                                f"<div style='background-color:#f1f8e9; border-radius:8px; margin-bottom:4px;'>"
                                f"<b style='color:#388e3c'>{nombre}</b>: <span style='color:#1976d2'>{valor}</span></div>",
                                unsafe_allow_html=True
                            )
                    pdf_bytes = generar_pdf(resultado)
                    st.download_button(
                        label="Descargar reporte en PDF",
                        data=pdf_bytes,
                        file_name="reporte_simulacion.pdf",
                        mime="application/pdf"
                    )
        except Exception as ex:
            st.error(f"Error en el cálculo: {ex}")
        col1, col2 = st.columns(2)
        if col1.button("Nuevo cálculo", key="asist_nuevo_paso6"):
            st.session_state.asist_paso = 1
            st.session_state.asist_modelo = None
            st.session_state.asist_lambda = None
            st.session_state.asist_mu = None
            st.session_state.asist_c = None
            st.session_state.asist_K = None
            st.session_state.asist_result = None
        if col2.button("⬅️ Volver al último dato", key="asist_volver_result_paso6"):
            if modelo in ["M/M/1"]:
                st.session_state.asist_paso = 3
            elif modelo == "M/M/c":
                st.session_state.asist_paso = 4
            elif modelo == "M/M/1/K":
                st.session_state.asist_paso = 5
            elif modelo == "M/M/c/K":
                st.session_state.asist_paso = 5

# -------- PESTAÑA 4: AYUDA Y CONCEPTOS --------
with tabs[3]:
    st.header("Ayuda y Conceptos de Teoría de Colas 📚")
    st.markdown("""
### 1. ¿Qué es un modelo de colas?
Un **sistema de colas** es un modelo matemático que describe el proceso de llegada de clientes a un sistema, su espera (si no hay servidores libres), su atención y salida del sistema.  
**Ejemplo:** personas haciendo fila para ser atendidas en un banco, llamadas a un call center, o procesos esperando CPU en un servidor.

---

### 2. Parámetros básicos

- **λ (lambda):** Tasa de llegada promedio (clientes por unidad de tiempo).  
    _Ejemplo: llegan 10 clientes por hora → λ = 10/hora_

- **μ (mu):** Tasa de servicio promedio de cada servidor (clientes por unidad de tiempo).  
    _Ejemplo: un servidor atiende 12 clientes por hora → μ = 12/hora_

- **c:** Número de servidores en paralelo.
- **K:** Capacidad máxima del sistema (servidores + clientes esperando).

---

### 3. Modelos clásicos

| Modelo        | Características                                | Ejemplo                                         |
|---------------|-----------------------------------------------|-------------------------------------------------|
| M/M/1         | 1 servidor, llegada y servicio exponencial, cola infinita | Cajero único en un banco                        |
| M/M/1/K       | 1 servidor, capacidad máxima K                | Ventanilla con solo 4 asientos                  |
| M/M/c         | c servidores, cola infinita                   | 3 médicos en una clínica                        |
| M/M/c/K       | c servidores, capacidad máxima K              | Call center con 5 líneas y 12 personas máximo   |

---

### 4. Glosario rápido

- **ρ (rho):** Utilización del sistema = λ / (c·μ)
- **Lq:** Promedio de clientes esperando (en la cola)
- **Ls:** Promedio de clientes en el sistema (cola + servicio)
- **Wq:** Tiempo promedio de espera en la cola
- **Ws:** Tiempo promedio total en el sistema

---

### 5. Tips y advertencias frecuentes

- Para que un sistema sea estable:  
    - En **M/M/1**: λ < μ  
    - En **M/M/c**: λ < c·μ
- Si λ ≥ μ (o c·μ), la cola crecerá indefinidamente (“el sistema se satura”).
- No uses valores negativos o cero para tasas ni para cantidades de servidores o capacidad.

---

### 6. Simulación de Monte Carlo en modelos de colas

La **simulación Monte Carlo** es una técnica que te permite **generar escenarios aleatorios** para estimar el comportamiento de sistemas complejos. En el contexto de colas, se utiliza para **ver cómo se comportarían las llegadas y los servicios si los datos fueran aleatorios** pero respetando ciertas distribuciones de probabilidad.

#### ¿Para qué sirve?

- Para explorar la **variabilidad** y la **incertidumbre** en los modelos.
- Para practicar cómo se verían los resultados si simularas 100, 1,000 o más llegadas y servicios.
- Para calcular promedios y desviaciones estándar de los tiempos y la cantidad de clientes en el sistema.

#### Distribuciones más comunes en colas:

- **Poisson:** Modela el número de llegadas por unidad de tiempo (por ejemplo, cuántos clientes llegan cada minuto).
    - En tu simulador, puedes pedir simular la llegada de clientes con esta distribución.
- **Exponencial:** Modela el tiempo entre llegadas o entre servicios (tiempo que pasa hasta que llega el siguiente cliente).
    - En tu simulador, puedes pedir simular estos tiempos aleatorios.

#### Ejemplo de interpretación en tu simulador:

- Si simulas 10 variables con 5 observaciones bajo Poisson (λ = 2), estás generando una tabla de 5 filas x 10 columnas donde cada número representa **cuántos clientes llegaron** en ese periodo de tiempo aleatorio.
- Si simulas con la distribución Exponencial, cada número es un **tiempo aleatorio de espera** entre llegadas o servicios.

#### ¿Cómo usar la pestaña de Monte Carlo en esta app?

1. **Selecciona la distribución:**  
   - Elige Poisson si quieres saber cuántos clientes llegan en ciertos intervalos.
   - Elige Exponencial si quieres saber el tiempo entre llegadas o servicios.
2. **Configura λ (lambda):**  
   - Para Poisson, representa el número promedio de llegadas por intervalo.
   - Para Exponencial, es la tasa promedio de ocurrencia.
3. **Elige cantidad de variables y observaciones:**  
   - Variables = número de columnas (puede simular diferentes servidores o periodos).
   - Observaciones = número de filas (simulaciones independientes).
4. **Haz clic en “Simular Monte Carlo” y revisa la tabla y estadísticas.**

#### Tips rápidos

- Puedes descargar los resultados en CSV para analizarlos en Excel, Google Sheets o Python.
- Si repites muchas veces la simulación (muchas observaciones), la **media** tenderá a acercarse a la λ que elegiste.
- Usar Monte Carlo es útil para ver la dispersión y para enseñar conceptos de aleatoriedad y probabilidad en sistemas de colas.

#### Recursos extra

- [Simulación Monte Carlo en Wikipedia](https://es.wikipedia.org/wiki/M%C3%A9todo_de_Montecarlo)
- [Video: ¿Qué es la simulación Monte Carlo? (YouTube)](https://www.youtube.com/results?search_query=simulacion+monte+carlo+explicacion)


 
""")
