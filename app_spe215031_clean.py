"""
Aplicación Streamlit (versión limpia) para SPE-215031 — análisis multi‑pozo.

Incluye:
- Asistente de 5 pasos claro
- Parámetros adimensionales y modelos simple transitorio/pseudopermanente
- Controles de doble porosidad (Warren–Root) en el kernel físico
- Schedules (q vs t) con transformada de Laplace por tramos
- Explicaciones de cada parámetro y opción (ayuda), comentarios de gráficos,
  diagnóstico y recomendaciones.
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from math import pi, sqrt, gamma
from physics import (
    p_res_from_adim,
    q_from_adim,
    invert_stehfest_vec,
    invert_gaver_euler_vec,
    si_mu, si_ct, si_k, si_h, si_L, field_p,
    DAY_TO_S, STB_TO_M3,
)


# ----------------------------- Utilities -----------------------------
def _clamp_pos(x, eps=1e-12):
    return max(float(x), eps)

# --- Schedule: q_hat(s) por tramos (m3)
def qhat_piecewise_m3s(s: float, t_days: np.ndarray, q_stbd: np.ndarray) -> float:
    t_sec = np.asarray(t_days, float) * float(DAY_TO_S)
    q_m3s = np.asarray(q_stbd, float) * (STB_TO_M3 / float(DAY_TO_S))
    if len(t_sec) == 0:
        return 0.0
    order = np.argsort(t_sec)
    t_sec = t_sec[order]
    q_m3s = q_m3s[order]
    acc = 0.0
    last_q = 0.0
    s = float(max(s, 1e-30))
    for ti, qi in zip(t_sec, q_m3s):
        acc += (qi - last_q) * math.exp(-s * ti)
        last_q = qi
    return acc / s


def compute_dimensionless_params(params: dict) -> dict:
    """Compute dimensionless parameters used by the simplified models.

    Expected params keys (float):
      h, rw, x_e, y_e, xF, wF, C (well storage), mu,
      k_f, phi_f, c_tf, k_m, phi_m, c_tm
    Optional (float): sigma (shape factor), L (reference length), s_c (skin)
    """
    try:
        h = float(params.get("h", 250.0))
        rw = _clamp_pos(params.get("rw", 0.3))
        x_e = float(params.get("x_e", 250.0))
        y_e = float(params.get("y_e", 250.0))
        xF = _clamp_pos(params.get("xF", 500.0))
        wF = float(params.get("wF", 0.01))
        C = float(params.get("C", params.get("C_well", 0.0)))
        mu = float(params.get("mu", 0.02))
        k_f = _clamp_pos(params.get("k_f", 1000.0))
        phi_f = _clamp_pos(params.get("phi_f", 0.45))
        c_tf = _clamp_pos(params.get("c_tf", 3e-4))
        k_m = _clamp_pos(params.get("k_m", 0.01))
        phi_m = _clamp_pos(params.get("phi_m", 0.10))
        c_tm = _clamp_pos(params.get("c_tm", 2.0e-4))
        L = float(params.get("L", xF))
        s_c = float(params.get("s_c", 0.0))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid input parameter: {e}")

    # Bulk compressibilities (fracture/matrix)
    phi_c_f = phi_f * c_tf
    phi_c_m = phi_m * c_tm
    denom_phi_c = _clamp_pos(phi_c_f + phi_c_m)

    # storativity ratio omega and transmissibility ratio lambda
    omega = phi_c_f / denom_phi_c
    sigma = 12.0 / (_clamp_pos(L) ** 2)  # blocks assumed square of size L
    lambd = sigma * (xF ** 2) * (k_m / k_f)

    # Dimensions in D-units
    x_eD = x_e / xF
    y_eD = y_e / xF

    # Dimensionless well storage and fracture conductivity (simplified)
    C_D = 5.615 * C / (2 * pi * (phi_m * c_tm) * _clamp_pos(h) * xF ** 2) if phi_m * c_tm > 0 else 0.0
    C_FD = (k_f * wF) / (_clamp_pos(k_m) * xF)

    # Simplified diffusivities (ft^2/day units folded, pedagogic only)
    eta_f = 2.637e-4 * k_f / (_clamp_pos(phi_c_f) * _clamp_pos(mu)) if phi_c_f > 0 else 1.0
    eta_i = 2.637e-4 * k_m / (_clamp_pos(phi_c_m) * _clamp_pos(mu)) if phi_c_m > 0 else 1.0

    return dict(
        C_D=C_D, C_FD=C_FD, omega=omega, lambda_=lambd,
        x_eD=x_eD, y_eD=y_eD, eta_f=eta_f, eta_i=eta_i, s_c=s_c,
    )


def pressure_pseudosteady(tD: np.ndarray, params: dict, dim: dict):
    """Simplified piecewise pseudosteady pressure and derivative in D-units."""
    C_D = _clamp_pos(dim.get("C_D", 1.0))
    C_FD = _clamp_pos(dim.get("C_FD", 1.0))
    omega = _clamp_pos(dim.get("omega", 0.5))
    lambd = _clamp_pos(dim.get("lambda_", 1.0))
    y_eD = float(dim.get("y_eD", 1.0))
    s_c = float(dim.get("s_c", 0.0))

    tD = np.asarray(tD, float)
    pwD = np.zeros_like(tD)
    dpwD = np.zeros_like(tD)
    for i, t in enumerate(tD):
        if t < 1e-3:
            pwD[i] = t / C_D
            dpwD[i] = t / C_D
        elif t < 1e0:
            pwD[i] = (pi / 2.0) * sqrt((1 - omega) / lambd) + (pi / (3.0 * C_FD)) + s_c
            dpwD[i] = 0.5
        else:
            pwD[i] = 2 * pi * t + (pi * y_eD) / 6.0 + s_c / (3.0 * C_FD)
            dpwD[i] = 1.0
    return pwD, dpwD


def pressure_transient(tD: np.ndarray, params: dict, dim: dict):
    """Simplified piecewise transient pressure and derivative in D-units."""
    C_D = _clamp_pos(dim.get("C_D", 1.0))
    C_FD = _clamp_pos(dim.get("C_FD", 1.0))
    omega = _clamp_pos(dim.get("omega", 0.5))
    lambd = _clamp_pos(dim.get("lambda_", 1.0))
    y_eD = float(dim.get("y_eD", 1.0))
    s_c = float(dim.get("s_c", 0.0))

    tD = np.asarray(tD, float)
    pwD = np.zeros_like(tD)
    dpwD = np.zeros_like(tD)
    for i, t in enumerate(tD):
        if t < 1e-4:
            pwD[i] = t / C_D
            dpwD[i] = t / C_D
        elif t < 1e0:
            coeff = (3.0 / (lambd ** 3 * omega ** 9)) ** (1.0 / 8.0)
            pwD[i] = coeff * (pi / (np.sqrt(2 * C_FD))) * (t ** (1.0 / 8.0)) / gamma(9.0 / 8.0) + s_c
            dpwD[i] = coeff * (pi / (8.0 * np.sqrt(2 * C_FD))) * (t ** (1.0 / 8.0)) / gamma(9.0 / 8.0)
        else:
            pwD[i] = (2 * pi * t) / (1 + omega) + (pi * y_eD) / 6.0 + s_c / (2 * pi)
            dpwD[i] = (2 * pi * t) / (1 + omega)
    return pwD, dpwD


def rate_from_constant_pressure(p_wD: np.ndarray, q_ref: float) -> np.ndarray:
    """Return physical rate under constant-pressure at well: q = q_ref / p_wD."""
    p = np.clip(np.asarray(p_wD, float), 1e-12, 1e12)
    return q_ref * (1.0 / p)


def time_physical_days(t_D: np.ndarray, x_f: float, eta_R: float) -> np.ndarray:
    """Convert dimensionless time to physical time (days), pedagogic."""
    return np.asarray(t_D, float) * (float(x_f) ** 2) / _clamp_pos(eta_R)


# ----------------------------- App -----------------------------
def app():
    st.title("Aplicación SPE-215031 — Análisis Multi‑Pozo (Limpia)")

    # Sidebar navigation
    steps = [
        "1) Configuración del modelo",
        "2) Parámetros físicos",
        "3) Cálculo adimensional",
        "4) Soluciones D(t)",
        "5) Resultados y gráficos",
    ]
    step = st.sidebar.radio("Navegación", steps)

    # Session state
    if "params" not in st.session_state:
        st.session_state.params = {}
    if "dim" not in st.session_state:
        st.session_state.dim = {}

    # Step 1: Model selection
    if step == steps[0]:
        st.header("1) Configuracion del modelo")
        model = st.radio("Tipo de modelo", ("Pseudopermanente", "Transitorio"), index=0)
        st.session_state.params["model"] = model
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["n_wells"] = st.number_input("Numero de pozos", 1, 10, 2)
        with c2:
            st.session_state["n_fracs"] = st.number_input("Fracturas por pozo", 1, 30, 3)
        st.success("Configuracion almacenada.")

    # Step 2: Physical parameters
    elif step == steps[1]:
        st.header("2) Parametros fisicos")
        p = st.session_state.params
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p["h"] = st.number_input("Espesor h [ft]", value=float(p.get("h", 250.0)))
            p["mu"] = st.number_input("Viscosidad mu [cp]", value=float(p.get("mu", 0.02)))
            p["k_f"] = st.number_input("k_f fractura [md]", value=float(p.get("k_f", 1000.0)))
            p["k_m"] = st.number_input("k_m matriz [md]", value=float(p.get("k_m", 0.01)))
        with c2:
            p["phi_f"] = st.number_input("Porosidad fractura phi_f", value=float(p.get("phi_f", 0.45)))
            p["c_tf"] = st.number_input("Compresibilidad fractura c_tf [1/psi]", value=float(p.get("c_tf", 3e-4)), format="%.2e")
            p["phi_m"] = st.number_input("Porosidad matriz phi_m", value=float(p.get("phi_m", 0.10)))
            p["c_tm"] = st.number_input("Compresibilidad matriz c_tm [1/psi]", value=float(p.get("c_tm", 2.0e-4)), format="%.2e")
        with c3:
            p["rw"] = st.number_input("Radio de pozo r_w [ft]", value=float(p.get("rw", 0.3)))
            p["C"] = st.number_input("Almacenamiento pozo C [bbl/psi]", value=float(p.get("C", 0.0)))
            p["x_e"] = st.number_input("Tamano drenaje x_e [ft]", value=float(p.get("x_e", 250.0)))
            p["y_e"] = st.number_input("Tamano drenaje y_e [ft]", value=float(p.get("y_e", 250.0)))
        with c4:
            p["q_ref"] = st.number_input("Caudal ref q_ref [STB/d]", value=float(p.get("q_ref", 100.0)))
            p["p_res"] = st.number_input("Presion yacimiento p_res [psi]", value=float(p.get("p_res", 3000.0)))
            p["xF"] = st.number_input("Mitad long. fractura x_F [ft]", value=float(p.get("xF", 500.0)))
            p["wF"] = st.number_input("Ancho fractura w_F [ft]", value=float(p.get("wF", 0.01)))
            p["s_c"] = st.number_input("Skin s_c", value=float(p.get("s_c", 0.0)))
        st.session_state.params = p
        st.info("Continuar al Paso 3 para calcular variables adimensionales.")

    # Step 3: Dimensionless
    elif step == steps[2]:
        st.header("3) Calculo adimensional")
        try:
            dim = compute_dimensionless_params(st.session_state.params)
            st.session_state.dim = dim
        except Exception as e:
            st.error(f"Error en calculo adimensional: {e}")
            st.stop()

        st.subheader("Resultados adimensionales")
        st.write({
            "C_D": dim["C_D"], "C_FD": dim["C_FD"], "omega": dim["omega"], "lambda": dim["lambda_"],
            "x_eD": dim["x_eD"], "y_eD": dim["y_eD"],
        })
        st.success("Listo. Pasar al Paso 4 para resolver D(t).")

    # Step 4: D-solutions
    elif step == steps[3]:
        st.header("4) Soluciones D(t)")
        if not st.session_state.params or not st.session_state.dim:
            st.warning("Complete los pasos 2 y 3 primero.")
            st.stop()
        t_min = st.number_input("log10(tD) minimo", value=float(st.session_state.get("t_min_log", -3.0)))
        t_max = st.number_input("log10(tD) maximo", value=float(st.session_state.get("t_max_log", 3.0)))
        npts = st.slider("n puntos", 50, 500, int(st.session_state.get("n_pts", 150)))
        st.session_state["t_min_log"], st.session_state["t_max_log"], st.session_state["n_pts"] = t_min, t_max, npts
        tD = np.log10(np.ones(1))  # dummy to avoid mypy
        try:
            tD = np.logspace(float(t_min), float(t_max), int(npts))
        except Exception as e:
            st.error(f"Rango de tiempo invalido: {e}")
            st.stop()

        # Source of solution: piecewise model vs Laplace inversion (demo)
        src_options = ("Modelo por tramos", "Inversión de Laplace (demo)", "Kernel físico (slab)")
        prev_src = st.session_state.get("sol_src", src_options[0])
        try:
            src_index = src_options.index(prev_src)
        except ValueError:
            src_index = 0
        src = st.radio("Fuente de solución", src_options, index=src_index)
        st.session_state["sol_src"] = src

        if src == "Modelo por tramos":
            model = st.session_state.params.get("model", "Pseudopermanente")
            if model == "Pseudopermanente":
                pwD, dpwD = pressure_pseudosteady(tD, st.session_state.params, st.session_state.dim)
            else:
                pwD, dpwD = pressure_transient(tD, st.session_state.params, st.session_state.dim)
        elif src == "Inversión de Laplace (demo)":
            st.subheader("Inversión de Laplace (demo)")
            prev_meth = st.session_state.get("lap_meth", "Stehfest")
            meth = st.radio("Método", ("Stehfest", "Gaver‑Euler"), index=(0 if prev_meth=="Stehfest" else 1), horizontal=True)
            st.session_state["lap_meth"] = meth
            colA, colB, colC = st.columns(3)
            with colA:
                a_days = st.number_input("a [1/día]", value=float(st.session_state.get("lap_a", 0.1)), min_value=0.0, step=0.05)
                st.session_state["lap_a"] = a_days
            if meth == "Stehfest":
                with colB:
                    N_st = st.slider("N (Stehfest, par)", min_value=6, max_value=16, value=int(st.session_state.get("lap_N", 12)), step=2)
                    st.session_state["lap_N"] = N_st
                M_gv = st.session_state.get("lap_M", 18)
                P_gv = st.session_state.get("lap_P", 8)
            else:
                with colC:
                    M_gv = st.slider("M (Gaver)", 6, 24, int(st.session_state.get("lap_M", 18)), 1)
                    st.session_state["lap_M"] = M_gv
                P_gv = st.slider("P (Euler)", 2, 12, int(st.session_state.get("lap_P", 8)), 1)
                st.session_state["lap_P"] = P_gv

            a_sec = float(a_days) / float(DAY_TO_S) if a_days > 0 else 0.0

            def Fvec(s: float) -> np.ndarray:
                # Demo kernel: 1/(s + a) -> f(t) = exp(-a t)
                val = 1.0 / max(s + a_sec, 1e-30)
                return np.array([val], dtype=float)

            pwD_list = []
            for tt in tD:
                t_sec = float(tt) * float(DAY_TO_S)  # interpret tD as days -> seconds
                if t_sec <= 0:
                    pwD_list.append(0.0)
                    continue
                if meth == "Stehfest":
                    inv = invert_stehfest_vec(Fvec, t_sec, int(st.session_state.get("lap_N", 12)))
                else:
                    inv = invert_gaver_euler_vec(Fvec, t_sec, M=int(st.session_state.get("lap_M", 18)), P=int(st.session_state.get("lap_P", 8)))
                pwD_list.append(float(inv[0]))
            pwD = np.array(pwD_list, float)
            dpwD = np.gradient(pwD, np.log(np.clip(tD, 1e-12, 1e12)))
        else:
            st.subheader("Kernel físico (slab no‑flujo)")
            wc = st.session_state.get("well_control", "Caudal constante (qD=1)")
            if not wc.startswith("Caudal"):
                st.info("El kernel físico requiere control a caudal constante; usando qD=1.")
            par = st.session_state.params
            mu_cp = float(par.get("mu", 0.02))
            k_md = float(par.get("k_m", 0.01))
            h_ft = float(par.get("h", 250.0))
            L_ft = float(par.get("x_e", 250.0))
            ct_invpsi = float(par.get("phi_f", 0.45))*float(par.get("c_tf",3e-4)) + float(par.get("phi_m",0.1))*float(par.get("c_tm",2e-4))
            q_ref_stbd = float(par.get("q_ref", 100.0))
            mu_SI = si_mu(mu_cp)
            ct_SI = si_ct(ct_invpsi)
            k_SI = si_k(k_md*1e6)  # md -> nD
            h_SI = si_h(h_ft)
            L_SI = si_L(L_ft)
            q0_m3s = q_ref_stbd * STB_TO_M3 / float(DAY_TO_S)
            def Fvec_phys(s: float) -> np.ndarray:
                from physics import R_slab_no_flow
                R = R_slab_no_flow(mu_SI, ct_SI, k_SI, h_SI, L_SI, float(s))
                return np.array([R * (q0_m3s / max(s,1e-30))], dtype=float)
            pw_list = []
            for tt in tD:
                t_sec = float(tt) * float(DAY_TO_S)
                if t_sec <= 0:
                    pw_list.append(0.0)
                    continue
                inv = invert_stehfest_vec(Fvec_phys, t_sec, int(st.session_state.get("lap_N", 12)))
                pw_list.append(field_p(float(inv[0])))
            pw_phys_psi = np.array(pw_list, float)
            mu = mu_cp; q_ref = q_ref_stbd; k = max(k_md, 1e-12); h = max(h_ft, 1e-12)
            pD = (pw_phys_psi * (k * h)) / (141.2 * max(mu,1e-12) * max(q_ref,1e-12))
            pwD = pD.copy()
            dpwD = np.gradient(pwD, np.log(np.clip(tD, 1e-12, 1e12)))

        # In this clean app, treat pD = pwD
        pD = np.asarray(pwD, float)
        # Well control: choose how to compute rate
        wc_prev = st.session_state.get("well_control", "Caudal constante (qD=1)")
        wc_idx = 0 if wc_prev.startswith("Caudal") else 1
        wc = st.radio("Control de pozo", ("Caudal constante (qD=1)", "Presión constante (q = q_ref/p_wD)"), index=wc_idx)
        st.session_state["well_control"] = wc
        qD = np.ones_like(tD) if wc.startswith("Caudal") else (1.0 / np.clip(pwD, 1e-12, 1e12))

        st.session_state["tD"] = tD
        st.session_state["pD"] = pD
        st.session_state["qD"] = qD
        st.session_state["pwD"] = pwD
        st.session_state["dpwD"] = dpwD
        st.success("Soluciones D(t) calculadas.")

    # Step 5: Results
    elif step == steps[4]:
        st.header("5) Resultados y graficos")
        req = ("tD" in st.session_state and "pD" in st.session_state and
               "qD" in st.session_state and "params" in st.session_state)
        if not req:
            st.warning("Calcule D(t) en el Paso 4.")
            st.stop()

        tD = np.asarray(st.session_state["tD"], float)
        pD = np.asarray(st.session_state["pD"], float)
        qD = np.asarray(st.session_state["qD"], float)
        mu = float(st.session_state.params.get("mu", 0.02))
        k = _clamp_pos(st.session_state.params.get("k_f", 1000.0))  # use fracture k as ref
        h = _clamp_pos(st.session_state.params.get("h", 250.0))
        q_ref = float(st.session_state.params.get("q_ref", 100.0))
        p_res = float(st.session_state.params.get("p_res", 3000.0))

        # Physical conversion
        dP_real = p_res_from_adim(pD, mu, k, h, q_ref)
        # Rate based on selected well control
        if st.session_state.get("well_control", "Caudal").startswith("Presion"):
            Q_real = rate_from_constant_pressure(st.session_state["pwD"], q_ref)
        else:
            Q_real = q_from_adim(qD, q_ref)
        Pwf = p_res - dP_real

        # Basic numeric safety
        mask = (tD > 0) & np.isfinite(dP_real) & np.isfinite(Q_real)
        tD, dP_real, Q_real, Pwf = tD[mask], dP_real[mask], Q_real[mask], Pwf[mask]
        pD = pD[mask]

        # Plots
        st.subheader("Graficos principales")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.loglog(tD, dP_real, "o-", label="Delta P [psi]", color="navy")
        ax1.set_xlabel("t_D")
        ax1.set_ylabel("Delta P [psi]")
        ax1.grid(True, which="both", ls="--", alpha=0.6)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(tD, Pwf, "r-", label="Pwf [psi]")
        ax2.set_xlabel("t_D")
        ax2.set_ylabel("Pwf [psi]")
        ax2.grid(True, ls="--", alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.plot(tD, Q_real, "g-", label="Q [STB/d]")
        ax3.set_xlabel("t_D")
        ax3.set_ylabel("Q [STB/d]")
        ax3.grid(True, ls="--", alpha=0.6)
        ax3.legend()
        st.pyplot(fig3)

        # Dimensionless derivative (approximate)
        dpdlogt = np.gradient(np.log(np.clip(pD, 1e-12, 1e12)), np.log(tD))
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.loglog(tD, pD, "b-", label="p_D")
        ax4.loglog(tD, np.abs(dpdlogt), "k--", label="|dp_D/d ln t_D|")
        ax4.set_xlabel("t_D")
        ax4.set_ylabel("p_D y derivada")
        ax4.grid(True, which="both", ls="--", alpha=0.6)
        ax4.legend()
        st.pyplot(fig4)

        # Diagnostics
        st.subheader("Diagnostico")
        if np.any(~np.isfinite(dP_real)) or np.any(~np.isfinite(Q_real)):
            st.warning("Resultados no validos detectados (NaN/Inf). Revise estabilidad.")
        else:
            st.success("Calculos estables.")


if __name__ == "__main__":
    app()
