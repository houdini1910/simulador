import streamlit as st
import numpy as np
from math import factorial
from fpdf import FPDF

# --- Diccionario de explicaciones ---
EXPLICACIONES = {
    "Modelo": "Tipo de sistema de colas utilizado",
    "lambda": "λ — Tasa de llegada (clientes por unidad de tiempo)",
    "mu": "μ — Tasa de servicio (clientes atendidos por servidor por unidad de tiempo)",
    "c": "c — Número de servidores",
    "K": "K — Capacidad máxima total del sistema (incluye en servicio y en cola)",
    "rho": "ρ — Utilización del sistema (porcentaje de tiempo ocupado)",
    "P0": "P₀ — Probabilidad de que no haya clientes en el sistema",
    "Lq": "Lq — Número promedio de clientes en la cola",
    "Ls": "Ls — Número promedio de clientes en el sistema (cola + servicio)",
    "Wq": "Wq — Tiempo promedio en cola (espera)",
    "Ws": "Ws — Tiempo promedio en el sistema (espera + servicio)",
    "lambda_eff": "λₑff — Tasa efectiva de llegada (clientes que realmente entran al sistema)",
    "Distribucion": "Distribución de probabilidad P(n) y acumulada para cada número de clientes en el sistema"
}

st.set_page_config(page_title="Simulador de Colas y Monte Carlo", layout="centered")

# ------ FUNCIONES DE CÁLCULO ------
def calcular_mm1(lmbda, mu):
    rho = lmbda / mu
    P0 = 1 - rho
    Lq = rho**2 / (1 - rho)
    Ls = Lq + rho
    Wq = Lq / lmbda
    Ws = Ls / lmbda
    return {"Modelo": "M/M/1", "lambda": lmbda, "mu": mu, "rho": rho, "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws}

def calcular_mmc(lmbda, mu, c):
    rho = lmbda / (c * mu)
    suma = sum((lmbda/mu)**n / factorial(n) for n in range(c))
    complemento = (lmbda/mu)**c / (factorial(c) * (1 - rho))
    P0 = 1 / (suma + complemento)
    Lq = (P0 * (lmbda/mu)**c * rho) / (factorial(c) * (1 - rho)**2)
    Ls = Lq + c * rho
    Wq = Lq / lmbda
    Ws = Wq + 1/mu
    return {"Modelo": f"M/M/{c}", "lambda": lmbda, "mu": mu, "c": c, "rho": rho, "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws}

def calcular_mmck(lmbda, mu, c, K):
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
            nombre = EXPLICACIONES.get(k, k)
            pdf.cell(0, 8, f"{nombre}: {v_str}", ln=1)
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

# ------ INTERFAZ PRINCIPAL ------
st.title("Simulador de Colas y Monte Carlo")

tabs = st.tabs(["Modelos de Colas", "Simulación Monte Carlo", "Asistente"])

# -------- PESTAÑA 1: MODELOS CLÁSICOS (igual que antes)
with tabs[0]:
    st.header("Simulación de Modelos de Colas")
    lmbda = st.number_input("λ (Tasa de llegada)", min_value=0.01, value=1.5, format="%.2f")
    mu = st.number_input("μ (Tasa de servicio)", min_value=0.01, value=2.5, format="%.2f")
    c = st.number_input("Cantidad de servidores (c)", min_value=1, step=1, value=1)

    limitar = st.checkbox("Limitar capacidad de la cola (M/M/c/K)")
    if limitar:
        K = st.number_input("Capacidad total K", min_value=int(c), step=1, value=int(c)+3)
    else:
        K = None

    resultado = None
    if st.button("Calcular"):
        try:
            if limitar:
                resultado = calcular_mmck(lmbda, mu, int(c), int(K))
            else:
                resultado = calcular_mm1(lmbda, mu) if int(c) == 1 else calcular_mmc(lmbda, mu, int(c))
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

# -------- PESTAÑA 2: MONTE CARLO (igual que antes)
with tabs[1]:
    st.header("Simulación de Monte Carlo (Poisson / Exponencial)")
    dist = st.radio("Selecciona la distribución", ["Poisson", "Exponencial"])
    lmbda_mc = st.number_input("λ (tasa promedio)", min_value=0.01, value=1.0, format="%.2f")
    n_vars = st.number_input("Cantidad de variables a simular", min_value=1, value=5)
    n_obs = st.number_input("Cantidad de observaciones", min_value=1, value=3)
    
    if st.button("Simular Monte Carlo"):
        try:
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

# -------- PESTAÑA 3: ASISTENTE MEJORADO PASO A PASO Y HUMANO

with tabs[2]:
    st.markdown("<h2 style='color:#0099ff'>Asistente Virtual 🤖</h2>", unsafe_allow_html=True)
    st.markdown("> **¡Resuelve tu problema de colas paso a paso!**\n")

    modelos = {
        "M/M/1": {
            "desc": "Un solo servidor, cola ilimitada.",
            "ej": "🧑‍💼 Ejemplo: Un cajero atendiendo en un banco sin límite de espera."
        },
        "M/M/1/K": {
            "desc": "Un solo servidor, capacidad limitada.",
            "ej": "🪑 Ejemplo: Sala de espera con solo 5 asientos."
        },
        "M/M/c": {
            "desc": "Varios servidores, cola ilimitada.",
            "ej": "🏢 Ejemplo: 3 médicos atendiendo pacientes en una clínica."
        },
        "M/M/c/K": {
            "desc": "Varios servidores, capacidad limitada.",
            "ej": "📞 Ejemplo: 5 líneas en un call center con máximo 10 personas en total."
        }
    }

    # Variables de control de paso
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
        st.subheader("1️⃣ Selecciona el tipo de modelo")
        for nombre, data in modelos.items():
            if st.button(f"Elegir {nombre}"):
                st.session_state.asist_modelo = nombre
                st.session_state.asist_paso = 2
                st.session_state.asist_lambda = None
                st.session_state.asist_mu = None
                st.session_state.asist_c = None
                st.session_state.asist_K = None
                st.session_state.asist_result = None
        for nombre, data in modelos.items():
            st.markdown(
                f"<div style='background-color:#e0f7fa; padding:12px; margin-bottom:6px; border-radius:8px;'>"
                f"<b style='color:#008080;'>{nombre}</b>: {data['desc']}<br>"
                f"<span style='color:#0099ff'>{data['ej']}</span></div>", unsafe_allow_html=True)
        st.info("💡 Elige el modelo que más se parece a tu situación real.")

    # Paso 2: Lambda
    elif st.session_state.asist_paso == 2:
        st.subheader("2️⃣ Ingresa la tasa de llegada λ")
        st.markdown("""
        <div style='color:#00695c;'>
        <b>¿Qué es λ?</b> Es el <b>número promedio de clientes</b> que llegan por unidad de tiempo.<br>
        <b>Ejemplo:</b> Si cada 2 minutos llegan 4 personas, entonces λ = 2 por minuto.
        <br><br>
        <b style='color:#ff9800;'>TIP:</b> Piensa: ¿cuántos clientes nuevos llegan en 1 hora? Divide por 60 si quieres el valor por minuto.
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("λ (tasa de llegada)", min_value=0.01, value=1.0, format="%.2f", key="asist_lambda")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_lambda"):
            st.session_state.asist_lambda = val
            st.session_state.asist_paso = 3
        if col2.button("⬅️ Volver", key="asist_volver_modelo"):
            st.session_state.asist_paso = 1

    # Paso 3: Mu
    elif st.session_state.asist_paso == 3:
        st.subheader("3️⃣ Ingresa la tasa de servicio μ")
        st.markdown("""
        <div style='color:#1a237e;'>
        <b>¿Qué es μ?</b> Es el <b>número promedio de clientes</b> que un servidor puede atender por unidad de tiempo.<br>
        <b>Ejemplo:</b> Si cada médico atiende 5 personas por hora, entonces μ = 5 por hora.<br>
        <b style='color:#ff9800;'>TIP:</b> Si hay varios servidores y atienden igual, pon la tasa individual aquí. Si no, elige el modelo c adecuado y lo sumas después.
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("μ (tasa de servicio)", min_value=0.01, value=2.0, format="%.2f", key="asist_mu")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_mu"):
            st.session_state.asist_mu = val
            if st.session_state.asist_modelo in ["M/M/c", "M/M/c/K"]:
                st.session_state.asist_paso = 4
            elif st.session_state.asist_modelo == "M/M/1/K":
                st.session_state.asist_paso = 5
            else:
                st.session_state.asist_paso = 6
        if col2.button("⬅️ Volver", key="asist_volver_lambda"):
            st.session_state.asist_paso = 2

    # Paso 4: c (servidores)
    elif st.session_state.asist_paso == 4:
        st.subheader("4️⃣ Ingresa la cantidad de servidores c")
        st.markdown("""
        <div style='color:#bf360c;'>
        <b>¿Qué es c?</b> Es el <b>número de servidores o puestos</b> que atienden simultáneamente.<br>
        <b>Ejemplo:</b> 4 ventanillas en un banco, c = 4.<br>
        <b style='color:#ff9800;'>TIP:</b> Si tienes un solo servidor, pon c=1 y te recomendamos usar M/M/1.
        </div>
        """, unsafe_allow_html=True)
        val = st.number_input("Cantidad de servidores (c)", min_value=1, value=2, step=1, key="asist_c")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_c"):
            st.session_state.asist_c = val
            if st.session_state.asist_modelo == "M/M/c/K":
                st.session_state.asist_paso = 5
            else:
                st.session_state.asist_paso = 6
        if col2.button("⬅️ Volver", key="asist_volver_mu"):
            st.session_state.asist_paso = 3

    # Paso 5: K (capacidad máxima)
    elif st.session_state.asist_paso == 5:
        st.subheader("5️⃣ Ingresa la capacidad máxima del sistema K")
        st.markdown("""
        <div style='color:#33691e;'>
        <b>¿Qué es K?</b> Es el <b>máximo número de personas</b> que pueden estar en el sistema (esperando + en servicio).<br>
        <b>Ejemplo:</b> 1 cajero y 5 sillas: K = 6.<br>
        <b style='color:#ff9800;'>TIP:</b> Si no hay límite, usa los modelos sin K.
        </div>
        """, unsafe_allow_html=True)
        min_c = int(st.session_state.asist_c) if st.session_state.asist_c else 1
        val = st.number_input("Capacidad total (K)", min_value=min_c, value=min_c + 3, step=1, key="asist_K")
        col1, col2 = st.columns(2)
        if col1.button("Siguiente ➡️", key="asist_siguiente_K"):
            st.session_state.asist_K = val
            st.session_state.asist_paso = 6
        if col2.button("⬅️ Volver", key="asist_volver_cK"):
            if st.session_state.asist_modelo == "M/M/1/K":
                st.session_state.asist_paso = 3
            else:
                st.session_state.asist_paso = 4

    # Paso 6: Resultados
    elif st.session_state.asist_paso == 6:
        st.markdown("<h3 style='color:#43a047'>Resultados y análisis 📊</h3>", unsafe_allow_html=True)
        modelo = st.session_state.asist_modelo
        lmbda = float(st.session_state.asist_lambda)
        mu = float(st.session_state.asist_mu)
        c = int(st.session_state.asist_c) if st.session_state.asist_c else 1
        K = int(st.session_state.asist_K) if st.session_state.asist_K else None
        resultado = None
        try:
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
        if col1.button("Nuevo cálculo", key="asist_nuevo"):
            st.session_state.asist_paso = 1
            st.session_state.asist_modelo = None
            st.session_state.asist_lambda = None
            st.session_state.asist_mu = None
            st.session_state.asist_c = None
            st.session_state.asist_K = None
            st.session_state.asist_result = None
        if col2.button("⬅️ Volver al último dato", key="asist_volver_result"):
            if modelo in ["M/M/1"]:
                st.session_state.asist_paso = 3
            elif modelo == "M/M/c":
                st.session_state.asist_paso = 4
            elif modelo == "M/M/1/K":
                st.session_state.asist_paso = 5
            elif modelo == "M/M/c/K":
                st.session_state.asist_paso = 5
