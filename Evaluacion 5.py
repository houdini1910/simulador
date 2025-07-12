import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from math import factorial
import os
import numpy as np  # <- Necesario para Monte Carlo

# Variable global para almacenar el último resultado
ultimo_resultado = None

# --- Cálculo de Modelos de Colas ---
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

# --- Generación de PDF ---
def generar_pdf(result, path=None):
    if path is None:
        path = filedialog.asksaveasfilename(defaultextension='.pdf')
    if not path:
        return None
    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(50, 780, "Reporte de Simulación de Colas")
    y = 750
    for k, v in result.items():
        if k == 'Distribución':
            c.drawString(50, y, "Distribución de Probabilidades:")
            y -= 20
            c.drawString(60, y, "n | P(n) | Acumulada")
            y -= 20
            for i, (p, ac) in enumerate(v):
                c.drawString(60, y, f"{i} | {round(p,4)} | {round(ac,4)}")
                y -= 15
        else:
            txt = round(v,4) if isinstance(v, float) else v
            c.drawString(50, y, f"{k}: {txt}")
            y -= 20
    c.save()
    return path

# --- Función para imprimir directamente ---
def imprimir_pdf():
    global ultimo_resultado
    if ultimo_resultado is None:
        messagebox.showwarning("Imprimir", "No hay resultado para imprimir.")
        return
    path = generar_pdf(ultimo_resultado)
    if path:
        try:
            os.startfile(path, "print")
        except Exception:
            messagebox.showerror("Error Imprimir", f"No se pudo imprimir. PDF guardado en:\n{path}")
        else:
            messagebox.showinfo("Imprimir", f"Enviado a impresora:\n{path}")

# --- GUI Principal ---
root = tk.Tk()
root.title("Simulador de Colas Avanzado")
root.geometry('520x670')
root.configure(padx=10, pady=10)

frame = ttk.Frame(root, padding=10)
frame.pack(fill='both', expand=True)
for i in range(14): frame.rowconfigure(i, weight=1)
for j in range(2): frame.columnconfigure(j, weight=1)

# Título
ttk.Label(frame, text="SIMULADOR DE COLAS AVANZADO", font=('Helvetica', 18, 'bold')) \
    .grid(row=0, column=0, columnspan=2, pady=(0,15))

# Modo de uso
ttk.Label(frame, text="Modo de Uso:", font=('Helvetica', 12)) \
    .grid(row=1, column=0, sticky='w', padx=5, pady=5)
modo = tk.StringVar(value='manual')
ttk.Radiobutton(frame, text='Manual', variable=modo, value='manual') \
    .grid(row=1, column=1, sticky='w', padx=5)
ttk.Radiobutton(frame, text='Asistente', variable=modo, value='asistente') \
    .grid(row=1, column=1, sticky='e', padx=5)

# Entrada de parámetros
widgets = [
    ("λ (llegada):", 2),
    ("μ (servicio):", 3),
    ("Servidores (c):", 4)
]
for text, row in widgets:
    ttk.Label(frame, text=text) \
        .grid(row=row, column=0, sticky='w', padx=5, pady=5)

ent_lambda = ttk.Entry(frame)
ent_lambda.grid(row=2, column=1, sticky='ew', padx=5)
ent_mu = ttk.Entry(frame)
ent_mu.grid(row=3, column=1, sticky='ew', padx=5)
ent_c = ttk.Spinbox(frame, from_=1, to=10)
ent_c.grid(row=4, column=1, sticky='ew', padx=5)

var_limite = tk.BooleanVar()
ttk.Checkbutton(frame, text="Limitar capacidad", variable=var_limite) \
    .grid(row=5, column=0, columnspan=2, pady=5)

ttk.Label(frame, text="K (si aplica):") \
    .grid(row=6, column=0, sticky='w', padx=5)
ent_K = ttk.Entry(frame)
ent_K.grid(row=6, column=1, sticky='ew', padx=5)

# Área de resultados
txt = tk.Text(frame, height=12)
txt.grid(row=7, column=0, columnspan=2, sticky='nsew', pady=10, padx=5)

# Funciones de cálculo y asistente
def calcular_manual():
    try:
        lmbda = float(ent_lambda.get())
        mu = float(ent_mu.get())
        c = int(ent_c.get())
        if var_limite.get():
            K = int(ent_K.get())
            res = calcular_mmck(lmbda, mu, c, K)
        else:
            res = calcular_mm1(lmbda, mu) if c == 1 else calcular_mmc(lmbda, mu, c)
        mostrar_resultado(res)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def iniciar_asistente():
    try:
        messagebox.showinfo("Asistente", "Bienvenido al asistente. Te guiaré paso a paso.")
        lmbda = simpledialog.askfloat("tasa llegada λ", "Ingresa la tasa de llegada:", minvalue=0.0)
        mu = simpledialog.askfloat("tasa servicio μ", "Ingresa la tasa de servicio:", minvalue=0.0)
        c = simpledialog.askinteger("Servidores c", "¿Cuántos servidores?", minvalue=1)
        limite = messagebox.askyesno("Límite", "¿Limitar capacidad de cola?")
        if limite:
            K = simpledialog.askinteger("K total", "Ingresa capacidad total:", minvalue=c)
            res = calcular_mmck(lmbda, mu, c, K)
        else:
            res = calcular_mm1(lmbda, mu) if c == 1 else calcular_mmc(lmbda, mu, c)
        mostrar_resultado(res)
        messagebox.showinfo("Asistente", "Cálculo completado.")
    except Exception as e:
        messagebox.showerror("Error Asistente", str(e))

def mostrar_resultado(res):
    global ultimo_resultado
    ultimo_resultado = res
    txt.delete('1.0', tk.END)
    for k, v in res.items():
        if k == 'Distribución':
            txt.insert(tk.END, "Distribución P(n) y acumulada:\n")
            for i, (p, ac) in enumerate(v):
                txt.insert(tk.END, f"n={i}: P={round(p,4)}, Ac={round(ac,4)}\n")
        else:
            txt.insert(tk.END, f"{k}: {round(v,4) if isinstance(v, float) else v}\n")

btn_calc = ttk.Button(frame, text="Calcular", command=calcular_manual)
btn_asist = ttk.Button(frame, text="Iniciar Asistente", command=iniciar_asistente)
btn_pdf = ttk.Button(frame, text="Guardar PDF", command=lambda: generar_pdf(ultimo_resultado))
btn_print = ttk.Button(frame, text="Imprimir", command=imprimir_pdf)

# Lógica de modos
def actualizar_modo(*args):
    if modo.get() == 'manual':
        btn_calc.grid(row=8, column=0, sticky='ew', padx=5, pady=5)
        btn_asist.grid_forget()
    else:
        btn_asist.grid(row=8, column=0, sticky='ew', padx=5, pady=5)
        btn_calc.grid_forget()
    btn_pdf.grid(row=8, column=1, sticky='ew', padx=5, pady=5)
    btn_print.grid(row=9, column=1, sticky='ew', padx=5, pady=5)

modo.trace_add('write', actualizar_modo)
actualizar_modo()

# ------- NUEVA SECCIÓN: Simulación de Monte Carlo Mejorada -------
def abrir_ventana_montecarlo():
    win = tk.Toplevel(root)
    win.title("Simulación de Monte Carlo")
    win.geometry('440x500')
    win.resizable(False, False)

    # Título
    ttk.Label(win, text="Simulación Monte Carlo\n(Poisson o Exponencial)", font=('Helvetica', 14, 'bold')).pack(pady=8)

    # Selección de distribución
    dist_var = tk.StringVar(value='poisson')
    frame_distrib = ttk.LabelFrame(win, text="Tipo de distribución")
    frame_distrib.pack(pady=8, padx=10, fill='x')
    ttk.Radiobutton(frame_distrib, text="Poisson", variable=dist_var, value='poisson').pack(side='left', padx=10, pady=6)
    ttk.Radiobutton(frame_distrib, text="Exponencial", variable=dist_var, value='exponencial').pack(side='left', padx=10, pady=6)

    # Parámetro lambda
    param_frame = ttk.Frame(win)
    param_frame.pack(pady=3, padx=10, fill='x')
    ttk.Label(param_frame, text="λ (tasa promedio):").pack(side='left', padx=3)
    param_entry = ttk.Entry(param_frame, width=10)
    param_entry.pack(side='left', padx=8, fill='x', expand=True)

    # Cantidad de variables a simular
    n_frame = ttk.Frame(win)
    n_frame.pack(pady=3, padx=10, fill='x')
    ttk.Label(n_frame, text="Cantidad de variables:").pack(side='left', padx=3)
    n_entry = ttk.Entry(n_frame, width=10)
    n_entry.pack(side='left', padx=8, fill='x', expand=True)

    # Cantidad de observaciones
    obs_frame = ttk.Frame(win)
    obs_frame.pack(pady=3, padx=10, fill='x')
    ttk.Label(obs_frame, text="Cantidad de observaciones:").pack(side='left', padx=3)
    obs_entry = ttk.Entry(obs_frame, width=10)
    obs_entry.pack(side='left', padx=8, fill='x', expand=True)

    # Área para mostrar resultados
    result_text = tk.Text(win, height=10, font=("Consolas", 10))
    result_text.pack(pady=8, padx=10, fill='both', expand=True)

    # Función para simular
    def simular():
        try:
            dist = dist_var.get()
            lmbda = float(param_entry.get())
            n_vars = int(n_entry.get())
            n_obs = int(obs_entry.get())

            if n_vars <= 0 or n_obs <= 0 or lmbda <= 0:
                raise ValueError("Todos los valores deben ser mayores a cero.")

            if dist == 'poisson':
                resultados = np.random.poisson(lmbda, size=(n_obs, n_vars))
            else:
                resultados = np.random.exponential(1/lmbda, size=(n_obs, n_vars))

            # Mostrar primeros resultados y estadísticos
            result_text.delete('1.0', tk.END)
            result_text.insert(tk.END, "Primeras 5 simulaciones:\n")
            for fila in resultados[:5]:
                fila_str = "  ".join(str(round(x, 4)) for x in fila)
                result_text.insert(tk.END, f"{fila_str}\n")
            result_text.insert(tk.END, f"\nMedia total: {np.mean(resultados):.4f}\n")
            result_text.insert(tk.END, f"Desviación estándar: {np.std(resultados):.4f}\n")
        except Exception as e:
            result_text.delete('1.0', tk.END)
            result_text.insert(tk.END, f"Error: {str(e)}")

    # Frame para los botones
    btns_frame = ttk.Frame(win)
    btns_frame.pack(pady=18)

    # Botón Simular 
    btn_simular = ttk.Button(btns_frame, text="Simular", command=simular)
    btn_simular.pack(side='left', padx=14, ipadx=24, ipady=6)

    # Botón Regresar
    btn_regresar = ttk.Button(btns_frame, text="Regresar al simulador de colas", command=win.destroy)
    btn_regresar.pack(side='left', padx=10, ipadx=8, ipady=6)

# En la ventana principal agrega un botón al final
btn_mc = ttk.Button(frame, text="Simulación Monte Carlo", command=abrir_ventana_montecarlo)
btn_mc.grid(row=12, column=0, columnspan=2, sticky='ew', padx=5, pady=10)

root.mainloop()
