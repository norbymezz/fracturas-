"""
Streamlit application for demonstrating the multi‑well interference model
described in SPE‑215031.  The app guides users through a series of
steps (similar to a wizard) to select the problem type, enter
reservoir and fracture properties, compute the key dimensionless
parameters and, finally, visualise a simple early‑time pressure and
derivative response.  The implementation here does not attempt to
replicate the full nested analytical solution presented in the paper
(which would require a fairly involved Laplace domain inversion), but
it provides a framework that can be extended for more detailed
calculations.  All equations and definitions are taken from the
publicly available SPE‑215031 publication【814652039279267†L184-L299】【814652039279267†L337-L475】.

This module can be executed with ``streamlit run app_spe215031.py``.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def compute_reference_properties(kf_R, phi_f_R, ct_f_R, phi_m_R, ct_m_R, x_f_R):
    """Compute the reference diffusivity and storativity used to
    normalise time and pressure.  This follows Eqs. (3)–(5) of the
    paper.  We assume 1D flow in a slab with total system storativity
    defined by the sum of fracture and matrix contributions.【814652039279267†L236-L299】

    Parameters
    ----------
    kf_R : float
        Reference fracture permeability (md).
    phi_f_R : float
        Reference fracture porosity (fraction).
    ct_f_R : float
        Reference fracture compressibility (psi⁻¹).
    phi_m_R : float
        Reference matrix porosity (fraction).
    ct_m_R : float
        Reference matrix compressibility (psi⁻¹).
    x_f_R : float
        Reference fracture half‑length (ft).

    Returns
    -------
    eta_R : float
        Reference diffusivity (ft²/hr).
    ct_total_R : float
        Reference total storativity ((psi⁻¹)).
    """
    # Use consistent units for diffusivity: multiply by conversion
    # factor 2 (Eq. 4) for field units (t in days).  In the paper the
    # diffusivity is given by eta = 2*k/(phi*c*mu).  Here we ignore
    # viscosity because the reference diffusivity is used only for
    # dimensionless time scaling.
    # total storativity of the reference system
    ct_total_R = phi_f_R * ct_f_R + phi_m_R * ct_m_R
    # diffusivity (Eq. 4).  We include the 2 factor to match field
    # units conversion (6.328e-3) – this constant would normally
    # involve viscosity; here we set mu=1 for simplicity.
    eta_R = 2 * kf_R / ct_total_R
    return eta_R, ct_total_R


def compute_dimensionless_parameters(params):
    """Compute the dimensionless variables defined in the paper for a
    single well with uniform fracture properties.  The returned
    dictionary contains C_D, C_FD, h_D, r_wD and other parameters.

    Parameters
    ----------
    params : dict
        Dictionary of input parameters containing at minimum the
        following keys:
        - h : formation thickness (ft)
        - r_w : wellbore radius (ft)
        - k_f : fracture permeability (md)
        - w_f : fracture width (ft)
        - x_f : fracture half‑length (ft)
        - k_fract_ref : reference fracture permeability (md)
        - phi_f_ref : reference fracture porosity (fraction)
        - ct_f_ref : reference fracture compressibility (psi⁻¹)
        - phi_m_ref : reference matrix porosity (fraction)
        - ct_m_ref : reference matrix compressibility (psi⁻¹)
        - C_well : wellbore storage coefficient (bbl/psi)

    Returns
    -------
    dict
        Dictionary of dimensionless parameters.
    """
    # unpack parameters for readability
    h = params["h"]
    r_w = params["r_w"]
    k_f = params["k_f"]
    w_f = params["w_f"]
    x_f = params["x_f"]
    C_well = params["C_well"]
    # reference properties
    kf_R = params["k_fract_ref"]
    phi_f_R = params["phi_f_ref"]
    ct_f_R = params["ct_f_ref"]
    phi_m_R = params["phi_m_ref"]
    ct_m_R = params["ct_m_ref"]

    # compute reference diffusivity and storativity
    eta_R, ct_total_R = compute_reference_properties(
        kf_R, phi_f_R, ct_f_R, phi_m_R, ct_m_R, x_f
    )

    # dimensionless formation thickness and wellbore radius (Eqs. 20–21)
    h_D = h / x_f
    r_wD = r_w / x_f

    # dimensionless fracture conductivity C_FD (Eq. 22)
    # Using k_f of the well and kf_R reference
    C_FD = (k_f * w_f) / (kf_R * x_f)

    # dimensionless wellbore storage coefficient C_D (Eq. 24).  We
    # convert bbl to ft³ (1 bbl = 5.615 ft³).  The factor 3 in the
    # paper is 3 = 5.615 for field units【814652039279267†L417-L441】.
    # The denominator includes the reference total storativity and
    # h*r_w^2.  We set mu=1 for simplicity because it cancels out when
    # computing dimensionless pressure later on.
    C_D = (5.615 * C_well) / (2 * np.pi * ct_total_R * h * r_w**2)

    return {
        "h_D": h_D,
        "r_wD": r_wD,
        "C_FD": C_FD,
        "C_D": C_D,
        "eta_R": eta_R,
        "ct_total_R": ct_total_R,
    }


def early_time_pressure(t_D, C_D):
    """Return dimensionless pressure and derivative for early time
    (linear) flow.  According to Table 3A/3B line 1 in the paper, the
    pressure and derivative both follow p_wD = t_D / C_D for early
    times【814652039279267†L236-L299】.  This simple function can be
    extended to include more time‑regime approximations.

    Parameters
    ----------
    t_D : array_like
        Dimensionless time.
    C_D : float
        Dimensionless wellbore storage coefficient.

    Returns
    -------
    p_wD : ndarray
        Dimensionless pressure.
    dp_wD : ndarray
        Dimensionless derivative of pressure with respect to log(t_D).
    """
    p_wD = t_D / C_D
    dp_wD = t_D / C_D
    return p_wD, dp_wD


def app():
    st.title("Modelo de interferencia multiposo – SPE‑215031")
    st.write(
        "Esta aplicación interactiva guía al usuario por los pasos "
        "para introducir parámetros de un conjunto de pozos horizontales "
        "fracturados y calcula algunas variables adimensionales básicas. "
        "El enfoque sigue la estructura de análisis presentada en el "
        "artículo SPE‑215031, aunque aquí se implementa una versión "
        "simplificada del modelo."
    )

    # initialise session state for wizard
    if "step" not in st.session_state:
        st.session_state.step = 1

    # Step 1 – elegir número de pozos y fracturas por pozo
    if st.session_state.step == 1:
        st.header("Paso 1: Selección del sistema de pozos")
        n_wells = st.number_input(
            "Número de pozos horizontales (n)", min_value=1, max_value=10, value=1, step=1
        )
        n_fractures = st.number_input(
            "Número de fracturas por pozo (n_F)", min_value=1, max_value=20, value=1, step=1
        )
        st.session_state.n_wells = n_wells
        st.session_state.n_fractures = n_fractures
        if st.button("Continuar al Paso 2"):
            st.session_state.step = 2

    # Step 2 – ingresar parámetros de yacimiento y fracturas
    elif st.session_state.step == 2:
        st.header("Paso 2: Parámetros de yacimiento y fracturas")
        st.write(
            "Introduzca valores representativos para todos los pozos. "
            "Estos parámetros se utilizan para calcular las variables adimensionales."
        )
        # Inputs for reservoir and fracture properties
        params = {}
        params["h"] = st.number_input("Espesor de la formación h (ft)", value=250.0)
        params["r_w"] = st.number_input("Radio del pozo r_w (ft)", value=0.3)
        params["k_f"] = st.number_input("Permeabilidad de fractura k_f (md)", value=1000.0)
        params["w_f"] = st.number_input("Ancho de fractura w_f (ft)", value=0.02)
        params["x_f"] = st.number_input("Medio long. de fractura x_f (ft)", value=250.0)
        params["C_well"] = st.number_input(
            "Coeficiente de almacenamiento del pozo C (bbl/psi)", value=0.0
        )
        st.subheader("Propiedades de referencia para el escalamiento")
        params["k_fract_ref"] = st.number_input(
            "Permeabilidad de referencia k_f,R (md)", value=1000.0
        )
        params["phi_f_ref"] = st.number_input(
            "Porosidad de fractura de referencia φ_f,R", value=0.05
        )
        params["ct_f_ref"] = st.number_input(
            "Compresibilidad de fractura de referencia c_t,f,R (psi⁻¹)", value=1e-4
        )
        params["phi_m_ref"] = st.number_input(
            "Porosidad de matriz de referencia φ_m,R", value=0.1
        )
        params["ct_m_ref"] = st.number_input(
            "Compresibilidad de matriz de referencia c_t,m,R (psi⁻¹)", value=1e-5
        )
        # Save params
        st.session_state.params = params
        if st.button("Continuar al Paso 3"):
            st.session_state.step = 3

    # Step 3 – calcular parámetros adimensionales
    elif st.session_state.step == 3:
        st.header("Paso 3: Cálculo de variables adimensionales")
        params = st.session_state.params
        dim_params = compute_dimensionless_parameters(params)
        st.session_state.dim_params = dim_params

        st.write("**Resultados de las variables adimensionales:**")
        st.write(
            f"Espesor adimensional h_D = {dim_params['h_D']:.3f}\n"
            f"Radio de pozo adimensional r_wD = {dim_params['r_wD']:.3f}\n"
            f"Conductividad de fractura adimensional C_FD = {dim_params['C_FD']:.3e}\n"
            f"Coeficiente de almacenamiento adimensional C_D = {dim_params['C_D']:.3e}"
        )
        if st.button("Continuar al Paso 4"):
            st.session_state.step = 4

    # Step 4 – resolver modelo simplificado
    elif st.session_state.step == 4:
        st.header("Paso 4: Resolución simplificada del modelo")
        st.write(
            "Se calculará una aproximación de presión y derivada adimensional en "
            "regímenes de tiempo tempranos usando p_wD = t_D / C_D. "
            "Esta expresión procede de la aproximación de flujo lineal temprano "
            "de las Tablas 3A y 3B del artículo【814652039279267†L236-L299】."
        )
        dim_params = st.session_state.dim_params
        C_D = dim_params["C_D"]
        # Generate dimensionless time vector (log spaced)
        t_D = np.logspace(-3, 3, 100)
        p_wD, dp_wD = early_time_pressure(t_D, C_D)
        st.session_state.results = (t_D, p_wD, dp_wD)
        if st.button("Continuar al Paso 5"):
            st.session_state.step = 5

    # Step 5 – visualización
    elif st.session_state.step == 5:
        st.header("Paso 5: Visualización de resultados")
        t_D, p_wD, dp_wD = st.session_state.results
        fig, ax = plt.subplots()
        ax.loglog(t_D, p_wD, label="Presión adimensional p_wD")
        ax.loglog(t_D, dp_wD, label="Derivada adimensional dp_wD/dln(t_D)", linestyle="--")
        ax.set_xlabel("Tiempo adimensional t_D")
        ax.set_ylabel("Magnitud adimensional")
        ax.legend()
        ax.grid(True, which="both", ls=":", lw=0.5)
        st.pyplot(fig)
        # Option to restart
        if st.button("Reiniciar" ):
            st.session_state.step = 1


if __name__ == "__main__":
    app()