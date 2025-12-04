"""
Aplicación Streamlit avanzada para el modelo multi‑pozo SPE‑215031.

Este asistente amplía la versión básica al proporcionar:

* Cálculo de variables adimensionales a partir de los parámetros de
  yacimiento y fracturas introducidos por el usuario.
* Visualización de las expresiones utilizadas, mostrando las
  fórmulas clave en LaTeX para que el usuario comprenda los
  cálculos intermedios.
* Cálculo de respuestas de presión adimensional en tres regímenes
  simplificados (temprano, intermedio y tardío) basados en las
  tablas 3A/3B del artículo.  Se incluye una derivada aproximada
  respecto al tiempo adimensional.
* Conversión de la presión adimensional a caída de presión física
  (DeltaP) empleando las definiciones de presión adimensional【814652039279267†L184-L199】.
* Generación de gráficos de caudal (en unidades físicas) y de
  DeltaP en función del tiempo.

La aplicación no implementa el algoritmo completo de inversión de
Laplace descrito en el artículo; en su lugar, utiliza
aproximaciones cerradas para ilustrar cómo se comportan los
regímenes de flujo típicos.  Este ejemplo pretende servir como
plantilla para desarrollos más avanzados y como material didáctico.

Ejecutar con ``streamlit run app_spe215031_adv.py``.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# Utilities for computing dimensionless parameters

def compute_reference_properties(kf_R, phi_f_R, ct_f_R, phi_m_R, ct_m_R):
    """Compute reference diffusivity (η_R) and total storativity (ct_total_R).

    Based on Eqs. (4)–(5), the reference diffusivity for the total
    system is defined by

    η_R = 2 k_f,R / [(φ_f,R c_t,f,R + φ_m,R c_t,m,R)]【814652039279267†L236-L299】.

    The conversion factor 2 is appropriate for field units.  Viscosidad
    no se incluye explícitamente porque los parámetros adimensionales
    se escalan con ella más adelante.
    """
    ct_total_R = phi_f_R * ct_f_R + phi_m_R * ct_m_R
    eta_R = 2.0 * kf_R / ct_total_R
    return eta_R, ct_total_R


def compute_dimensionless_params(h, r_w, k_f, w_f, x_f, C_well,
                                 kf_R, phi_f_R, ct_f_R, phi_m_R, ct_m_R):
    """Compute dimensionless parameters h_D, r_wD, C_FD and C_D.

    Parameters correspond to Eqs. (20)–(24)【814652039279267†L337-L441】.  The
    wellbore storage coefficient is converted from bbl/psi to ft³/psi
    using the factor 5.615.  The viscosity µ will be applied later in
    the physical conversion of pressures.
    """
    # Reference properties and diffusivity
    eta_R, ct_total_R = compute_reference_properties(kf_R, phi_f_R, ct_f_R,
                                                     phi_m_R, ct_m_R)
    # Dimensionless thickness and radius
    h_D = h / x_f
    r_wD = r_w / x_f
    # Dimensionless fracture conductivity (Eq. 22)
    C_FD = (k_f * w_f) / (kf_R * x_f)
    # Dimensionless wellbore storage coefficient (Eq. 24)
    # 5.615 ft³/bbl conversion; denominator 2π ct_total_R h r_w²
    C_D = (5.615 * C_well) / (2 * math.pi * ct_total_R * h * r_w ** 2)
    return {
        'h_D': h_D,
        'r_wD': r_wD,
        'C_FD': C_FD,
        'C_D': C_D,
        'eta_R': eta_R,
        'ct_total_R': ct_total_R,
    }


def pressure_regimes(t_D, C_D, C_FD, regime='early'):
    """Return dimensionless pressure and derivative for specified flow regime.

    This helper implements simplified expressions drawn from
    Tables 3A/3B.  Only approximate forms are included here to
    illustrate early, intermediate and late regimes.  In a complete
    implementation, the appropriate expression would depend on the
    storativity ratio ω and the transmissibility ratio λ; those are
    omitted here for brevity.

    Parameters
    ----------
    t_D : ndarray
        Dimensionless time values.
    C_D : float
        Dimensionless wellbore storage coefficient.
    C_FD : float
        Dimensionless fracture conductivity.
    regime : {'early', 'intermediate', 'late'}
        Flow regime for which to compute the pressure response.

    Returns
    -------
    (p_wD, dp_wD) : tuple of ndarrays
        Dimensionless pressure and derivative with respect to log(t_D).
    """
    if regime == 'early':
        # linear flow (pseudosteady or transient) – p = t_D / C_D【814652039279267†L236-L299】
        p_wD = t_D / C_D
        dp_wD = t_D / C_D
    elif regime == 'intermediate':
        # approximate expression for intermediate times (fracture linear to
        # pseudosteady).  We use p ∝ t_D^(1/4), scaled by fracture
        # conductivity, as per rows 5–6 in Table 3A.
        coeff = (math.pi / (4.0 * math.sqrt(2.0 * C_FD)))
        p_wD = coeff * t_D ** 0.25
        # derivative: d/dln(t) = (1/4) p
        dp_wD = 0.25 * p_wD
    else:
        # late time: boundary dominated or pseudosteady state.  Use
        # p = 2π t_D (cf. row 9 in Table 3A)【814652039279267†L337-L475】
        p_wD = 2.0 * math.pi * t_D
        dp_wD = 2.0 * math.pi * t_D
    return p_wD, dp_wD


def convert_to_physical(p_wD, k_f, h, q_R, B, mu):
    """Convert dimensionless pressure to physical pressure drop (DeltaP).

    From the definition of dimensionless pressure (Eq. 1),

        p_wD = (k_f h) / (q_R B μ) * (p_i - p_w).

    Therefore Δp = p_w = (q_R B μ / (k_f h)) * p_wD【814652039279267†L184-L199】.

    Parameters
    ----------
    p_wD : array_like
        Dimensionless pressure values.
    k_f : float
        Fracture permeability (md).
    h : float
        Formation thickness (ft).
    q_R : float
        Reference rate (STB/d or Mscf/d).
    B : float
        Factor de volumen de formación (bbl/STB).  Use 1.0 for gas.
    mu : float
        Viscosidad del fluido (cp).

    Returns
    -------
    ndarray
        Physical pressure drop (psi).
    """
    factor = (q_R * B * mu) / (k_f * h)
    return factor * np.asarray(p_wD)


def convert_to_time(t_D, x_f, eta_R):
    """Convert dimensionless time to physical time (days).

    Using Eq. (3), t_D = (eta_R / x_f²) t ⇒ t = t_D * x_f² / eta_R.
    """
    return t_D * (x_f ** 2) / eta_R


# -----------------------------------------------------------------------------
# Streamlit application

def app():
    st.title("Modelo multi‑pozo SPE‑215031 – versión avanzada")
    st.markdown(
        "Esta aplicación amplía la versión básica para mostrar las fórmulas "
        "y cálculos intermedios y generar gráficos de caudal y caída de "
        "presión (Δp).  Utiliza aproximaciones analíticas de los regímenes "
        "temprano, intermedio y tardío de las Tablas 3A/3B del artículo "
        "*Pressure‑ and Rate‑Transient Model for an Array of Interfering "
        "Fractured Horizontal Wells in Unconventional Reservoirs* (SPE 215031)."
    )

    if 'step' not in st.session_state:
        st.session_state.step = 1
    # Sidebar navigation: provide a way to jump between steps
    steps_list = [
        "Paso 1", "Paso 2", "Paso 3", "Paso 4", "Paso 5", "Matricial y Laplace"
    ]
    sidebar_selection = st.sidebar.radio(
        "Navegación", steps_list, index=st.session_state.step - 1
    )
    # Update step based on sidebar selection
    selected_index = steps_list.index(sidebar_selection) + 1
    if selected_index != st.session_state.step:
        st.session_state.step = selected_index

    # Step 1: modelo y configuración
    if st.session_state.step == 1:
        st.header("Paso 1: Configuración del sistema")
        st.write("Seleccione el número de pozos y fracturas por pozo. Actualmente todas las fracturas se suponen idénticas.")
        n_wells = st.number_input("Número de pozos horizontales n", min_value=1, max_value=10, value=1, step=1)
        n_fractures = st.number_input("Número de fracturas por pozo n_F", min_value=1, max_value=20, value=1, step=1)
        st.session_state.n_wells = n_wells
        st.session_state.n_fractures = n_fractures
        if st.button("Continuar al Paso 2"):
            st.session_state.step = 2
        # Permitir retroceder si se seleccionó vía sidebar

    # Step 2: entrada de parámetros
    elif st.session_state.step == 2:
        st.header("Paso 2: Parámetros de entrada")
        st.write("Introduzca parámetros físicos del yacimiento, fractura y escala de referencia.")
        # Formación y pozo
        h = st.number_input("Espesor de formación h (ft)", value=250.0)
        r_w = st.number_input("Radio del pozo r_w (ft)", value=0.3)
        k_f = st.number_input("Permeabilidad de fractura k_f (md)", value=1000.0)
        w_f = st.number_input("Ancho de fractura w_f (ft)", value=0.02)
        x_f = st.number_input("Medio largo de fractura x_f (ft)", value=250.0)
        C_well = st.number_input("Coeficiente de almacenamiento C (bbl/psi)", value=0.0)
        # Propiedades de referencia
        st.subheader("Propiedades de referencia para el escalamiento")
        kf_R = st.number_input("k_f,R (md)", value=1000.0)
        phi_f_R = st.number_input("φ_f,R", value=0.05)
        ct_f_R = st.number_input("c_t,f,R (psi⁻¹)", value=1e-4)
        phi_m_R = st.number_input("φ_m,R", value=0.1)
        ct_m_R = st.number_input("c_t,m,R (psi⁻¹)", value=1e-5)
        # Parámetros de conversión a unidades físicas
        st.subheader("Parámetros para cálculo de Δp y caudal físico")
        q_R = st.number_input("Tasa de referencia q_R (STB/d o Mscf/d)", value=1000.0)
        B = st.number_input("Factor de volumen de formación B (bbl/STB)", value=1.0)
        mu = st.number_input("Viscosidad del fluido μ (cp)", value=1.0)
        # Guardar en session
        st.session_state.inputs = {
            'h': h, 'r_w': r_w, 'k_f': k_f, 'w_f': w_f, 'x_f': x_f, 'C_well': C_well,
            'kf_R': kf_R, 'phi_f_R': phi_f_R, 'ct_f_R': ct_f_R,
            'phi_m_R': phi_m_R, 'ct_m_R': ct_m_R,
            'q_R': q_R, 'B': B, 'mu': mu,
        }
        if st.button("Continuar al Paso 3"):
            st.session_state.step = 3
        if st.button("Volver al Paso 1"):
            st.session_state.step = 1

    # Step 3: cálculo de variables adimensionales
    elif st.session_state.step == 3:
        st.header("Paso 3: Cálculo de variables adimensionales")
        inp = st.session_state.inputs
        dim = compute_dimensionless_params(
            inp['h'], inp['r_w'], inp['k_f'], inp['w_f'], inp['x_f'], inp['C_well'],
            inp['kf_R'], inp['phi_f_R'], inp['ct_f_R'], inp['phi_m_R'], inp['ct_m_R']
        )
        st.session_state.dim = dim
        # Mostrar resultados y fórmulas
        st.markdown("**Fórmulas usadas:**")
        st.latex(r"h_D = \frac{h}{x_f}")
        st.latex(r"r_{wD} = \frac{r_w}{x_f}")
        st.latex(r"C_{FD} = \frac{k_f w_f}{k_{f,R} x_f}")
        st.latex(r"C_D = \frac{5.615 C}{2\pi \phi_{f+m,R} c_{t,f+m,R} \, h \, r_w^2}")
        st.markdown("**Valores calculados:**")
        st.write(
            f"Espesor adimensional h_D = {dim['h_D']:.3f}"\
            f"\nRadio adimensional r_wD = {dim['r_wD']:.3f}"\
            f"\nConductividad de fractura C_FD = {dim['C_FD']:.3e}"\
            f"\nCoeficiente de almacenamiento C_D = {dim['C_D']:.3e}"
        )
        if st.button("Continuar al Paso 4"):
            st.session_state.step = 4
        if st.button("Volver al Paso 2"):
            st.session_state.step = 2

    # Step 4: resolver regímenes de flujo
    elif st.session_state.step == 4:
        st.header("Paso 4: Cálculo de respuestas de presión adimensional")
        dim = st.session_state.dim
        # Elegir régimen
        regime = st.selectbox(
            "Seleccione el régimen de flujo a visualizar",
            ('early', 'intermediate', 'late'),
            format_func=lambda x: {'early': 'Temprano (flujo lineal)', 'intermediate': 'Intermedio (combinado)', 'late': 'Tardío (pseudosteady)'}[x]
        )
        # Tiempo adimensional log‑espaciado
        t_D = np.logspace(-3, 3, 100)
        p_wD, dp_wD = pressure_regimes(t_D, dim['C_D'], dim['C_FD'], regime)
        st.session_state.calc = {
            't_D': t_D, 'p_wD': p_wD, 'dp_wD': dp_wD, 'regime': regime
        }
        st.markdown(
            "**Expresión usada para p_{wD}:**" + (r"\quad p_{wD} = \frac{t_D}{C_D}" if regime == 'early' else (r"\quad p_{wD} \propto t_D^{1/4}" if regime == 'intermediate' else r"\quad p_{wD} = 2\pi t_D"))
        )
        if st.button("Continuar al Paso 5"):
            st.session_state.step = 5
        if st.button("Volver al Paso 3"):
            st.session_state.step = 3

    # Step 5: conversión a unidades físicas y gráficas
    elif st.session_state.step == 5:
        st.header("Paso 5: Conversión a unidades físicas y gráficos")
        inp = st.session_state.inputs
        dim = st.session_state.dim
        calc = st.session_state.calc
        t_D = calc['t_D']
        p_wD = calc['p_wD']
        dp_wD = calc['dp_wD']
        # convertir tiempo y presión a unidades físicas
        t_days = convert_to_time(t_D, inp['x_f'], dim['eta_R'])
        delta_p = convert_to_physical(p_wD, inp['k_f'], inp['h'], inp['q_R'], inp['B'], inp['mu'])
        # asumir q_D = 1 (producción a tasa constante) → q = q_R
        q_physical = np.full_like(t_days, inp['q_R'])
        # Mostrar curvas
        fig1, ax1 = plt.subplots()
        ax1.loglog(t_days, delta_p, label='Δp (psi)')
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel('Caída de presión Δp (psi)')
        ax1.legend()
        ax1.grid(True, which='both', ls=':')
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots()
        ax2.plot(t_days, q_physical)
        ax2.set_xscale('log')
        ax2.set_xlabel('Tiempo (días)')
        ax2.set_ylabel('Caudal q (STB/d o Mscf/d)')
        ax2.set_title('Caudal constante en el modelo simplificado')
        st.pyplot(fig2)
        # Mostrar tabla de algunos valores
        st.markdown("### Valores numéricos (muestra)")
        st.write("Los valores a continuación corresponden a una selección de tiempos.")
        sample_idx = [0, 20, 40, 60, 80, 99]
        sample_data = {
            't (días)': t_days[sample_idx],
            't_D': t_D[sample_idx],
            'p_wD': p_wD[sample_idx],
            'Δp (psi)': delta_p[sample_idx],
            'q (físico)': q_physical[sample_idx],
        }
        st.dataframe(sample_data)
        if st.button("Volver al Paso 4"):
            st.session_state.step = 4
        if st.button("Reiniciar"):
            st.session_state.step = 1

    # Step 6: Matricial y transformadas de Laplace
    elif st.session_state.step == 6:
        st.header("Paso 6: Operaciones matriciales y transformadas de Laplace")
        st.markdown(
            "En la formulación general del modelo, las soluciones de cada "
            "segmento de flujo (reservorios interiores y exteriores, y fracturas) "
            "se expresan en el dominio de Laplace y luego se acoplan en forma de "
            "una ecuación matricial:\n\n"
            "$$\mathbf{P}(s) = \mathbf{A}(s)\,\mathbf{Q}(s),$$"
            "donde $\mathbf{P}(s)$ y $\mathbf{Q}(s)$ son vectores de presiones y caudales "
            "en el dominio de Laplace para cada pozo, y $\mathbf{A}(s)$ es una matriz de "
            "coeficientes que depende de las propiedades de los bloques adyacentes. "
            "Para un sistema de dos pozos, la matriz tiene la forma:\n\n"
            "$$\mathbf{A}(s) = \\begin{pmatrix}A_{11}(s) & A_{12}(s) \ A_{21}(s) & A_{22}(s)\end{pmatrix}.$$"
        )
        st.markdown(
            "Las entradas $A_{ij}(s)$ se calculan analíticamente utilizando las "
            "soluciones de flujo en cada bloque y aplicando condiciones de "
            "continuidad de presión y flujo en las interfaces. Por ejemplo, "
            "$A_{12}(s)$ representa la influencia del pozo 2 sobre la presión en el pozo 1."
        )
        st.markdown(
            "Una vez formulada la ecuación matricial, la solución en el tiempo se "
            "obtiene invirtiendo la transformada de Laplace. Un método numérico "
            "común para esta inversión es el algoritmo de **Stehfest**. Dado "
            "un término $F(s)$ en el dominio de Laplace, la aproximación de "
            "Stehfest para recuperar la función $f(t)$ se expresa como:\n\n"
            "$$f(t) \\approx \\frac{\ln 2}{t}\sum_{k=1}^{N}\!\\frac{V_k}{k}\,F\!\left(k\\frac{\ln 2}{t}\\right),$$"
            "donde $V_k$ son coeficientes precomputados y $N$ es un número par que "
            "controla la precisión."
        )
        st.markdown(
            "En esta aplicación no implementamos la inversión de Laplace completa. "
            "Sin embargo, puede ampliarse incorporando los módulos ``math_laplace`` "
            "y ``math_dev`` del proyecto, que contienen funciones para armar la "
            "matriz de coeficientes y aplicar la inversión numérica."
        )
        if st.button("Volver al Paso 5"):
            st.session_state.step = 5


if __name__ == "__main__":
    app()