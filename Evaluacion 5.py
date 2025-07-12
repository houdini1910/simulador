import streamlit as st
import numpy as np
from math import factorial

st.set_page_config(page_title="Simulador de Colas y Monte Carlo", layout="centered")

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
    - Puedes copiar los resultados y guardarlos como desees.
    - ¡Si tienes dudas, revisa este manual o consulta a tu profesor!

    ---
    """)

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

st.title("Simulador de Colas y Simulación de Monte Carlo")

tabs = st.tabs(["Modelos de Colas", "Simulación Monte Carlo", "Ayuda / Manual"])

with tabs[0]:
    st.header("Simulación de Modelos de Colas")
    st.write("Elige y configura tu modelo:")

    lmbda = st.number_input("λ (Tasa de llegada)", min_value=0.01, value=1.5, format="%.2f")
    mu = st.number_input("μ (Tasa de servicio)", min_value=0.01, value=2.5, format="%.2f")
    c = st.number_input("Cantidad de servidores (c)", min_value=1, step=1, value=1)

    limitar = st.checkbox("Limitar capacidad de la cola (M/M/c/K)")
    if limitar:
        K = st.number_input("Capacidad total K", min_value=int(c), step=1, value=int(c)+3)
    else:
        K = None

    if st.button("Calcular"):
        try:
            if limitar:
                res = calcular_mmck(lmbda, mu, int(c), int(K))
            else:
                res = calcular_mm1(lmbda, mu) if int(c) == 1 else calcular_mmc(lmbda, mu, int(c))
            st.success("¡Cálculo realizado con éxito!")
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
        except Exception as ex:
            st.error(f"Error: {ex}")

with tabs[2]:
    mostrar_manual()
