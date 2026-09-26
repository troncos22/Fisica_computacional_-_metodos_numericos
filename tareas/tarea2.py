import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solver Diferencias Finitas", layout="wide")
st.title("Análisis de Esquemas Numéricos")

tab1, tab2, tab3, tab4 = st.tabs([
    "Ecuación de Advección Lineal", 
    "Cuerda No Lineal (DEIM)", 
    "Análisis Teórico: Advección",
    "Análisis Teórico: Cuerda No Lineal"
])

# ---------------------------------------------------------
# PESTAÑA 1: Advección
# ---------------------------------------------------------
with tab1:
    st.header("Comparación de Viscosidad Artificial")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        nu_choice = st.radio("Número de Courant (ν)", ["0.5", "0.8", "1.0"])
    
    df_adv = pd.read_csv(f"adveccion_nu_{nu_choice}.csv")
    
    with col2:
        fig_adv, ax_adv = plt.subplots(figsize=(10, 4))
        ax_adv.plot(df_adv['x'], df_adv['exact'], label='Solución Exacta', color='black', linestyle='--')
        ax_adv.plot(df_adv['x'], df_adv['lax'], label='Método Lax')
        ax_adv.plot(df_adv['x'], df_adv['upwind'], label='Método Upwind')
        
        ax_adv.set_title(f"Perfil de onda (Courant = {nu_choice})")
        ax_adv.set_xlabel("x")
        ax_adv.set_ylabel("u(x, t)")
        ax_adv.legend()
        ax_adv.grid(True, alpha=0.3)
        st.pyplot(fig_adv)
        
    with st.expander("Ver inestabilidad de FTCS"):
        st.warning("Los valores de FTCS divergen masivamente (escalas de 1e39 a 1e43), confirmando su inestabilidad incondicional.")
        st.line_chart(df_adv.set_index('x')['ftcs'])

# ---------------------------------------------------------
# PESTAÑA 2: Cuerda No Lineal
# ---------------------------------------------------------
with tab2:
    st.header("Dinámica Transversal y Reducción de Orden")
    
    col3, col4 = st.columns([1, 4])
    with col3:
        a_choice = st.radio("Amplitud Inicial (A)", ["0.1", "0.5", "1.0"])
        
    df_cuerda = pd.read_csv(f"cuerda_comparacion_A_{a_choice}.csv")
    
    with col4:
        fig_cuerda, ax_cuerda = plt.subplots(figsize=(10, 4))
        ax_cuerda.plot(df_cuerda['x'], df_cuerda['y_lineal'], label='Onda Lineal (Referencia)', color='grey', linestyle='--')
        ax_cuerda.plot(df_cuerda['x'], df_cuerda['y_fom'], label='No Lineal (FOM)', linewidth=2)
        ax_cuerda.plot(df_cuerda['x'], df_cuerda['y_deim'], label='No Lineal (DEIM)', linestyle=':', linewidth=2, color='red')
        
        ax_cuerda.set_title(f"Deformación de Crestas (Amplitud = {a_choice})")
        ax_cuerda.set_xlabel("x")
        ax_cuerda.set_ylabel("y(x, t)")
        ax_cuerda.legend()
        ax_cuerda.grid(True, alpha=0.3)
        st.pyplot(fig_cuerda)
        
        st.info("Observa cómo la curva DEIM se superpone casi perfectamente a la FOM evaluando la no linealidad en solo 10 nodos.")

# ---------------------------------------------------------
# PESTAÑA 3: Análisis Teórico Advección
# ---------------------------------------------------------
with tab3:
    st.header("Fundamentos Físicos y Matemáticos")
    
    st.subheader("1. Inestabilidad Incondicional de FTCS")
    st.markdown(r"""
    El análisis del factor de amplificación de Von Neumann demuestra que el esquema FTCS es incondicionalmente inestable para la advección pura. Al proponer una solución de la forma \(u_j^n = G^n e^{i k j \Delta x}\), el factor de amplificación es:
$$G = 1 - i \nu \sin(k \Delta x)$$Su magnitud al cuadrado es \(|G|^2 = 1 + \nu^2 \sin^2(k \Delta x)\). Al ser estrictamente mayor a 1 para \(\nu > 0\), el esquema inyecta energía espuria sistemáticamente (difusión numérica negativa).
""")

st.subheader("2. Viscosidad Artificial")
st.markdown(r"""
* **Courant óptimo (\(\nu = 1.0\)):** El término de difusión se anula y la matriz de diferencias finitas actúa como un operador de traslación exacta.
* **Difusión espuria (\(\nu < 1.0\)):** El método resuelve una ecuación equivalente \(u_t + c u_x = D u_{xx}\) con \(D = \frac{c \Delta x}{2} (1 - \nu)\). A menor \(\nu\), mayor es el aplastamiento del perfil de onda.
""")

st.subheader("3. Dispersión Numérica (Lax)")
st.markdown(r"""
El método de Lax introduce errores dispersivos de tercer orden \(\mathcal{O}(\Delta x^2)\). Diferentes números de onda viajan a distintas velocidades de fase, separando los componentes de Fourier y creando oscilaciones en la estela de la perturbación.
""")
#---------------------------------------------------------PESTAÑA 4: Análisis Teórico Cuerda---------------------------------------------------------with tab4:st.header("Dinámica Transversal: Análisis Físico y DEIM")st.subheader("1. Velocidad Efectiva y Retraso de Fase")
st.markdown(r"""
La ecuación diferencial modula la aceleración en función de la pendiente de la cuerda:
$$\frac{\partial^2 y}{\partial t^2} = c^2 \left[ 1 + \left( \frac{\partial y}{\partial x} \right)^2 \right]^{-1/2} \frac{\partial^2 y}{\partial x^2}$$Reescribiendo la ecuación en su forma de onda canónica $y_{tt} = c_{eff}^2 y_{xx}$, la velocidad de fase efectiva resulta ser:$$c_{eff} = c \left[ 1 + \left( \frac{\partial y}{\partial x} \right)^2 \right]^{-1/4}$$Cualquier deformación espacial \(\left(\frac{\partial y}{\partial x} \neq 0\right)\) provoca que \(c_{eff} < c\).
""")

st.subheader("2. Efecto de la Amplitud (A)")
st.markdown(r"""
* **Régimen cuasi-lineal (A = 0.1):** Los gradientes espaciales tienden a cero. La onda viaja a la velocidad \(c\) de la onda lineal clásica, logrando superposición exacta.
* **Régimen de retraso (A = 0.5 y 1.0):** Los flancos de la campana adquieren gradientes pronunciados que desploman localmente la \(c_{eff}\). Mientras la cresta (pendiente nula) intenta avanzar a velocidad \(c\), los costados de alta pendiente la frenan. Esto produce el retraso espacial (*lag*) visible respecto a la referencia lineal.
""")

st.subheader("3. Precisión de DEIM")
st.markdown(r"""
La solución interpolada empíricamente (DEIM) logra reconstruir el estado FOM sin desvíos. La matriz de proyección oblicua aísla exitosamente los nodos críticos de la malla espacial, reduciendo el costo de evaluación de la no linealidad de \(\mathcal{O}(N)\) a \(\mathcal{O}(m)\) conservando la exactitud física.
""")
