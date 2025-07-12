import streamlit as st
import numpy as np
from math import factorial
from fpdf import FPDF
import io

st.set_page_config(page_title="Simulador de Colas y Monte Carlo", layout="centered")

# ------ FUNCIONES DE CÁLCULO ------
def calcular_mm1(lmbda, mu):
    rho = lmbda / mu
    P0 = 1 - rho
    Lq = rho**2 / (1 - rho)
    Ls = Lq + rho
    Wq = Lq / lmbda
    Ws = Ls / lmbda
    return {"Modelo": "M/M/1", "λ": lmbda, "μ": mu, "ρ": rho, "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws}

def calcular_mmc(lmbda, mu, c):
    rho = lmbda / (c * mu)
    suma = sum((lmbda/mu)**n / factorial(n) for n in range(c))
    complemento = (lmbda/mu)**c / (factorial(c) * (1 - rho))
    P0 = 1 / (suma + complemento)
    Lq = (P0 * (lmbda/mu)**c * rho) / (factorial(c) * (1 - rho)**2)
    Ls = Lq + c * rho
    Wq = Lq / lmbda
    Ws = Wq + 1/mu
    return {"Modelo": f"M/M/{c}", "λ": lmbda, "μ": mu, "c": c, "ρ": rho, "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws}

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
    return {"Modelo": f"M/M/{c}/{K}", "λ": lmbda, "μ": mu, "c": c, "K": K, "ρ": rho_s / c,
            "P0": P0, "Lq": Lq, "Ls": Ls, "Wq": Wq, "Ws": Ws,
            "λ_eff": lambda_eff, "Distribución": list(zip(P, cumul))}

# ------ PDF GENERATION ------
def generar_pdf(result_dict, filename="reporte_simulacion.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Reporte de Simulación de Colas", ln=1, align='C')
    pdf.ln(10)
    for k, v in result_dict.items():
        if k == "Distribución":
            pdf.cell(200, 8, "Distribución P(n) y acumulada:", ln=1)
            for i, (p, ac) in enumerate(v):
                pdf.cell(200, 8, f"n={i}: P={round(p,4)}, Acumulada={round(ac,4)}", ln=1)
        else:
            pdf.cell(200, 8, f"{k}: {round(v,4) if isinstance(v, float) else v}", ln=1)
    b = pdf.output(dest='S').encode('latin1')
    return b

# ------ MANUAL ------
def mostrar_manual():
    st.markdown("""
    ### Manual de Usuario

    **1. Modelos de Colas (M/M/1, M/M/c, M/M/c/K):**
    - Introduce los valores de λ (tasa de llegada), μ (tasa de servicio), y servidores (c).
    - Si tu modelo tiene capacidad limitada, marca la casilla y coloca el valor de K.
    - Haz clic en "Calcular" para obtener resultados, o usa el "Asistente" para guía paso a paso.

    **2. Simulación Monte Carlo (Poisson/Exponencial):**
    - Ve a la pestaña "Simulación Monte Carlo".
    - Elige la distribución (Poisson o Exponencial).
    - Ingresa el parámetro λ, la cantidad de variables y de observaciones.
    - Pulsa "Simular" para ver los resultados, media y desviación estándar.

    **3. Reportes:**
    - Puedes descargar tus resultados como PDF.
    - ¡Si tienes dudas, revisa este manual o consulta a tu profesor!

    ---
    """)

# ------ INTERFAZ PRINCIPAL ------
st.title("Simulador de Colas y Monte Carlo")

tabs = st.tabs(["Modelos de Colas", "Simulación Monte Carlo", "Asistente", "Ayuda / Manual"])

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
                if k == "Distribución":
                    st.write("**Distribución P(n) y acumulada:**")
                    st.table(
                        [{"n": i, "P(n)": round(p,4), "Acumulada": round(ac,4)} for i, (p, ac) in enumerate(v)]
                    )
                else:
                    st.write(f"**{k}:** {round(v, 4) if isinstance(v, float) else v}")
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
            # Descarga simple
            st.download_button(
                label="Descargar resultados (CSV)",
                data=io.StringIO('\n'.join([','.join([str(x) for x in fila]) for fila in resultados])),
                file_name="resultados_montecarlo.csv",
                mime="text/csv"
            )
        except Exception as ex:
            st.error(f"Error: {ex}")

# -------- PESTAÑA 3: ASISTENTE (Wizard paso a paso)
with tabs[2]:
    st.header("Asistente Virtual - Modelos de Colas")

    paso = st.session_state.get('paso', 1)
    if 'paso' not in st.session_state:
        st.session_state.paso = 1

    if paso == 1:
        lmbda_asist = st.number_input("Paso 1: Ingresa la tasa de llegada λ", min_value=0.01, value=1.0, key="lmbda_asist")
        if st.button("Siguiente", key="btn1"):
            st.session_state.paso = 2

    if paso >= 2:
        mu_asist = st.number_input("Paso 2: Ingresa la tasa de servicio μ", min_value=0.01, value=2.0, key="mu_asist")
        if st.button("Siguiente", key="btn2"):
            st.session_state.paso = 3

    if paso >= 3:
        c_asist = st.number_input("Paso 3: Ingresa la cantidad de servidores c", min_value=1, value=1, step=1, key="c_asist")
        limitar_asist = st.checkbox("¿Limitar capacidad de la cola?", key="limitar_asist")
        if limitar_asist:
            k_asist = st.number_input("Capacidad total K", min_value=int(c_asist), value=int(c_asist)+3, step=1, key="k_asist")
        else:
            k_asist = None
        if st.button("Calcular Resultado (Asistente)", key="btn3"):
            try:
                if limitar_asist:
                    res = calcular_mmck(st.session_state.lmbda_asist, st.session_state.mu_asist, int(st.session_state.c_asist), int(st.session_state.k_asist))
                else:
                    res = calcular_mm1(st.session_state.lmbda_asist, st.session_state.mu_asist) if int(st.session_state.c_asist) == 1 else calcular_mmc(st.session_state.lmbda_asist, st.session_state.mu_asist, int(st.session_state.c_asist))
                st.success("¡Cálculo realizado!")
                for k, v in res.items():
                    if k == "Distribución":
                        st.write("**Distribución P(n) y acumulada:**")
                        st.table(
                            [{"n": i, "P(n)": round(p,4), "Acumulada": round(ac,4)} for i, (p, ac) in enumerate(v)]
                        )
                    else:
                        st.write(f"**{k}:** {round(v, 4) if isinstance(v, float) else v}")
            except Exception as ex:
                st.error(f"Error: {ex}")

    if st.button("Reiniciar Asistente"):
        st.session_state.paso = 1
        st.experimental_rerun()

# -------- PESTAÑA 4: AYUDA
with tabs[3]:
    mostrar_manual()
