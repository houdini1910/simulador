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
    "rho": "ρ — Utilización del sistema",
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

# -------- PESTAÑA 1: MODELOS CLÁSICOS
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

# -------- PESTAÑA 2: MONTE CARLO
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

# -------- PESTAÑA 3: ASISTENTE (wizard con selección de modelo)
with tabs[2]:
    st.header("Asistente Virtual - Modelos de Colas")
    st.markdown("Sigue los pasos para resolver tu problema de colas.")

    modelos = {
        "M/M/1": {"desc": "Un servidor, cola infinita", "ej": "Ideal para: Un cajero, un médico, un mecánico"},
        "M/M/1/K": {"desc": "Un servidor, capacidad limitada", "ej": "Ideal para: Sala de espera con asientos limitados"},
        "M/M/c": {"desc": "Múltiples servidores, cola infinita", "ej": "Ideal para: Múltiples cajeros, varios médicos"},
        "M/M/c/K": {"desc": "Múltiples servidores, capacidad limitada", "ej": "Ideal para: Call center con líneas limitadas"}
    }

    if 'paso' not in st.session_state:
        st.session_state.paso = 1
    if 'modelo_asist' not in st.session_state:
        st.session_state.modelo_asist = None

    if st.session_state.paso == 1:
        st.subheader("Paso 1: Selecciona el tipo de modelo")
        cols = st.columns(2)
        seleccion = None
        with cols[0]:
            if st.button("M/M/1"):
                seleccion = "M/M/1"
        with cols[1]:
            if st.button("M/M/1/K"):
                seleccion = "M/M/1/K"
        with cols[0]:
            if st.button("M/M/c"):
                seleccion = "M/M/c"
        with cols[1]:
            if st.button("M/M/c/K"):
                seleccion = "M/M/c/K"
        if seleccion:
            st.session_state.modelo_asist = seleccion
            st.session_state.paso = 2
        for k, v in modelos.items():
            st.write(f"**{k}** — {v['desc']}")
            st.caption(v["ej"])
    
    # Paso 2: Pedir parámetros según modelo
    if st.session_state.paso == 2:
        modelo = st.session_state.modelo_asist
        st.success(f"Modelo seleccionado: {modelo}")
        lmbda = st.number_input("λ (Tasa de llegada)", min_value=0.01, value=1.0, format="%.2f", key="lmbda_asistente")
        mu = st.number_input("μ (Tasa de servicio)", min_value=0.01, value=2.0, format="%.2f", key="mu_asistente")
        c = 1
        K = None
        if modelo in ["M/M/c", "M/M/c/K"]:
            c = st.number_input("Cantidad de servidores (c)", min_value=1, value=2, step=1, key="c_asistente")
        if modelo in ["M/M/1/K", "M/M/c/K"]:
            K = st.number_input("Capacidad total (K)", min_value=int(c), value=int(c)+3, step=1, key="k_asistente")

        if st.button("Calcular resultado", key="btn_asistente_calc"):
            try:
                if modelo == "M/M/1":
                    res = calcular_mm1(lmbda, mu)
                elif modelo == "M/M/c":
                    res = calcular_mmc(lmbda, mu, int(c))
                elif modelo == "M/M/1/K":
                    res = calcular_mmck(lmbda, mu, 1, int(K))
                elif modelo == "M/M/c/K":
                    res = calcular_mmck(lmbda, mu, int(c), int(K))
                else:
                    res = {}
                st.session_state.resultado_asistente = res
                st.session_state.paso = 3
            except Exception as ex:
                st.error(f"Error: {ex}")

        if st.button("Volver al paso anterior"):
            st.session_state.paso = 1

    # Paso 3: Mostrar resultados SOLO si existe resultado_asistente
    if st.session_state.paso == 3 and "resultado_asistente" in st.session_state:
        res = st.session_state.resultado_asistente
        st.success("¡Cálculo realizado!")
        for k, v in res.items():
            nombre = EXPLICACIONES.get(k, k)
            if k == "Distribucion":
                st.markdown(f"**{nombre}:**")
                st.table(
                    [{"n": i, "P(n)": round(p,4), "Acumulada": round(ac,4)} for i, (p, ac) in enumerate(v)]
                )
            else:
                valor = f"{v:.4f}" if isinstance(v, float) else v
                st.markdown(f"**{nombre}:** {valor}")

        if st.button("Realizar otro cálculo"):
            st.session_state.paso = 1
            st.session_state.modelo_asist = None
            del st.session_state["resultado_asistente"]
    elif st.session_state.paso == 3:
        st.warning("No hay resultados calculados. Por favor, vuelve a calcular un modelo.")
        if st.button("Volver a empezar"):
            st.session_state.paso = 1
            st.session_state.modelo_asist = None
            if "resultado_asistente" in st.session_state:
                del st.session_state.resultado_asistente
