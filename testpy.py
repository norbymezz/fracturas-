"""
Aplicación Streamlit — SPE-215031 (versión UTF‑8, completa)

- 5 pasos con ayudas en cada parámetro/opción
- Doble porosidad (Warren–Root) en kernel físico
- Schedules q(t) por tramos con q_hat(s)
- Inversión de Laplace (Stehfest / Gaver–Euler)
- Comentarios de gráficos, diagnóstico y recomendaciones
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


def _clamp_pos(x, eps=1e-12):
    return max(float(x), eps)


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
        raise ValueError(f"Parámetro inválido: {e}")

    phi_c_f = phi_f * c_tf
    phi_c_m = phi_m * c_tm
    denom_phi_c = _clamp_pos(phi_c_f + phi_c_m)
    omega = phi_c_f / denom_phi_c
    sigma = 12.0 / (_clamp_pos(L) ** 2)
    lambd = sigma * (xF ** 2) * (k_m / k_f)
    x_eD = x_e / xF
    y_eD = y_e / xF
    C_D = 5.615 * C / (2 * pi * (phi_m * c_tm) * _clamp_pos(h) * xF ** 2) if phi_m * c_tm > 0 else 0.0
    C_FD = (k_f * wF) / (_clamp_pos(k_m) * xF)
    eta_f = 2.637e-4 * k_f / (_clamp_pos(phi_c_f) * _clamp_pos(mu)) if phi_c_f > 0 else 1.0
    eta_i = 2.637e-4 * k_m / (_clamp_pos(phi_c_m) * _clamp_pos(mu)) if phi_c_m > 0 else 1.0
    return dict(C_D=C_D, C_FD=C_FD, omega=omega, lambda_=lambd,
                x_eD=x_eD, y_eD=y_eD, eta_f=eta_f, eta_i=eta_i, s_c=s_c)


def pressure_pseudosteady(tD: np.ndarray, params: dict, dim: dict):
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


def app():
    st.title("SPE‑215031 — Análisis Multi‑Pozo (UTF‑8)")

    steps = [
        "1) Configuración del modelo",
        "2) Parámetros físicos",
        "3) Cálculo adimensional",
        "4) Soluciones D(t)",
        "5) Resultados y gráficos",
    ]
    step = st.sidebar.radio("Navegación", steps)

    if "params" not in st.session_state:
        st.session_state.params = {}
    if "dim" not in st.session_state:
        st.session_state.dim = {}

    if step == steps[0]:
        st.header("1) Configuración del modelo")
        model_default = st.session_state.params.get("model", "Pseudopermanente")
        model = st.radio("Tipo de modelo", ("Pseudopermanente", "Transitorio"),
                         index=(0 if model_default == "Pseudopermanente" else 1),
                         help="Selecciona el régimen: pseudopermanente o transitorio.")
        st.session_state.params["model"] = model
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["n_wells"] = st.number_input("Número de pozos", 1, 10, int(st.session_state.get("n_wells", 2)))
        with c2:
            st.session_state["n_fracs"] = st.number_input("Fracturas por pozo", 1, 30, int(st.session_state.get("n_fracs", 3)))
        st.success("Configuración almacenada.")
# ----------------------------------------------------------
# Paso 1: Configuración del modelo (interactivo con figura)
# ----------------------------------------------------------
elif step == steps[0]:
    st.header("1️⃣ Configuración del modelo con visualización interactiva")

    st.markdown("""
    Ajustá los parámetros de permeabilidad de cada región del modelo trilineal.
    La figura mostrará el dominio correspondiente resaltado según el parámetro editado.
    """)

    import plotly.graph_objects as go

    # --- Imagen base ---
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="5f903beb-1384-4863-8a21-c110cd0ffd14.png",
            xref="x", yref="y",
            x=0, y=10, sizex=10, sizey=10, sizing="stretch", opacity=1, layer="below"
        )
    )

    # --- Sliders de parámetros ---
    col1, col2, col3 = st.columns(3)
    with col1:
        kF = st.slider("k_F (Hydraulic Fracture)", 1e-3, 1e3, 100.0, format="%.3f")
    with col2:
        kI = st.slider("k_I (Inner Reservoir)", 1e-6, 1e2, 10.0, format="%.6f")
    with col3:
        kO = st.slider("k_O (Outer Reservoir)", 1e-6, 1e2, 1.0, format="%.6f")

    # --- Guardar en sesión ---
    st.session_state.update({"kF": kF, "kI": kI, "kO": kO})

    # --- Posiciones de cada dominio ---
    param_coords = {
        "k_O": (7.5, 7.5),
        "k_I": (4.5, 5.0),
        "k_F": (2.0, 5.0),
    }

    # --- Determinar cuál slider fue modificado ---
    selected_param = st.radio(
        "Seleccioná un dominio para resaltar:",
        options=["k_F", "k_I", "k_O"],
        horizontal=True,
        help="El dominio seleccionado se resaltará en la figura."
    )

    # --- Dibujar círculos ---
    for param, (x, y) in param_coords.items():
        color = "red" if param == selected_param else "gray"
        size = 0.5 if param == selected_param else 0.3
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=x - size, y0=y - size, x1=x + size, y1=y + size,
            line=dict(color=color, width=3 if param == selected_param else 1),
        )
        fig.add_annotation(
            x=x, y=y,
            text=f"{param} = {st.session_state[param]:.4g} md",
            showarrow=False,
            font=dict(color=color, size=12)
        )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=850, height=450,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info("""
    **Leyenda:**
    - `k_F`: Permeabilidad de la fractura hidráulica  
    - `k_I`: Permeabilidad del reservorio interno (SRV o fracturado naturalmente)  
    - `k_O`: Permeabilidad del reservorio externo  
    """)

    st.success("Parámetros almacenados y visualizados correctamente ✅")

    elif step == steps[1]:
        st.header("2) Parámetros físicos")
        p = st.session_state.params
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p["h"] = st.number_input("Espesor h [ft]", value=float(p.get("h", 250.0)))
            p["mu"] = st.number_input("Viscosidad μ [cp]", value=float(p.get("mu", 0.02)))
            p["k_f"] = st.number_input("Permeabilidad fractura k_f [md]", value=float(p.get("k_f", 1000.0)))
            p["k_m"] = st.number_input("Permeabilidad matriz k_m [md]", value=float(p.get("k_m", 0.01)))
        with c2:
            p["phi_f"] = st.number_input("Porosidad fractura φ_f", value=float(p.get("phi_f", 0.45)))
            p["c_tf"] = st.number_input("Compresibilidad fractura c_tf [1/psi]", value=float(p.get("c_tf", 3e-4)), format="%.2e")
            p["phi_m"] = st.number_input("Porosidad matriz φ_m", value=float(p.get("phi_m", 0.10)))
            p["c_tm"] = st.number_input("Compresibilidad matriz c_tm [1/psi]", value=float(p.get("c_tm", 2.0e-4)), format="%.2e")
        with c3:
            p["rw"] = st.number_input("Radio de pozo r_w [ft]", value=float(p.get("rw", 0.3)))
            p["C"] = st.number_input("Almacenamiento de pozo C [bbl/psi]", value=float(p.get("C", 0.0)))
            p["x_e"] = st.number_input("Tamaño de drenaje x_e [ft]", value=float(p.get("x_e", 250.0)))
            p["y_e"] = st.number_input("Tamaño de drenaje y_e [ft]", value=float(p.get("y_e", 250.0)))
        with c4:
            p["q_ref"] = st.number_input("Caudal de referencia q_ref [STB/d]", value=float(p.get("q_ref", 100.0)))
            p["p_res"] = st.number_input("Presión de yacimiento p_res [psi]", value=float(p.get("p_res", 3000.0)))
            p["xF"] = st.number_input("Mitad de longitud de fractura x_F [ft]", value=float(p.get("xF", 500.0)))
            p["wF"] = st.number_input("Ancho de fractura w_F [ft]", value=float(p.get("wF", 0.01)))
            p["s_c"] = st.number_input("Skin s_c", value=float(p.get("s_c", 0.0)))
        st.session_state.params = p
        st.info("Pasá al Paso 3 para calcular variables adimensionales.")

    elif step == steps[2]:
        st.header("3) Cálculo adimensional")
        try:
            dim = compute_dimensionless_params(st.session_state.params)
            st.session_state.dim = dim
        except Exception as e:
            st.error(f"Error en cálculo adimensional: {e}")
            st.stop()
        st.subheader("Resultados adimensionales")
        st.write({k: st.session_state.dim[k] for k in ["C_D","C_FD","omega","lambda_","x_eD","y_eD"]})
        st.success("Listo. Paso 4 para resolver D(t).")

    elif step == steps[3]:
        st.header("4) Soluciones D(t)")
        if not st.session_state.params or not st.session_state.dim:
            st.warning("Completá los pasos 2 y 3 primero.")
            st.stop()
        t_min = st.number_input("log10(tD) mínimo", value=float(st.session_state.get("t_min_log", -3.0)))
        t_max = st.number_input("log10(tD) máximo", value=float(st.session_state.get("t_max_log", 3.0)))
        npts = st.slider("Número de puntos", 50, 500, int(st.session_state.get("n_pts", 150)))
        st.session_state["t_min_log"], st.session_state["t_max_log"], st.session_state["n_pts"] = t_min, t_max, npts

        st.subheader("Schedule de producción (q vs t)")
        if "sched" not in st.session_state:
            q0 = float(st.session_state.params.get("q_ref", 100.0))
            st.session_state.sched = pd.DataFrame({"t_dias": [0.0], "q_STB_d": [q0]})
        st.session_state.sched = st.data_editor(
            st.session_state.sched, use_container_width=True, num_rows="dynamic",
            column_config={
                "t_dias": st.column_config.NumberColumn("t [días]"),
                "q_STB_d": st.column_config.NumberColumn("q [STB/d]"),
            }
        )
        st.download_button("Descargar schedule (CSV)", st.session_state.sched.to_csv(index=False).encode("utf-8"),
                           "schedule.csv", "text/csv")

        try:
            tD = np.logspace(float(t_min), float(t_max), int(npts))
        except Exception as e:
            st.error(f"Rango inválido: {e}")
            st.stop()

        src = st.radio("Fuente de solución", ("Modelo por tramos", "Inversión de Laplace (demo)", "Kernel físico (slab)"), index=0)
        if src == "Modelo por tramos":
            if st.session_state.params.get("model", "Pseudopermanente") == "Pseudopermanente":
                pwD, dpwD = pressure_pseudosteady(tD, st.session_state.params, st.session_state.dim)
            else:
                pwD, dpwD = pressure_transient(tD, st.session_state.params, st.session_state.dim)
        elif src == "Inversión de Laplace (demo)":
            meth = st.radio("Método", ("Stehfest", "Gaver‑Euler"), index=0, horizontal=True)
            a_days = st.number_input("a [1/día]", value=0.1, min_value=0.0, step=0.05)
            a_sec = float(a_days) / float(DAY_TO_S) if a_days > 0 else 0.0
            if meth == "Stehfest":
                N_st = st.slider("N (Stehfest, par)", 6, 16, 12, 2)
            else:
                M_gv = st.slider("M (Gaver)", 6, 24, 18, 1)
                P_gv = st.slider("P (Euler)", 2, 12, 8, 1)
            def Fvec(s: float) -> np.ndarray:
                val = 1.0 / max(s + a_sec, 1e-30)
                return np.array([val], dtype=float)
            pwD = []
            for tt in tD:
                t_sec = float(tt) * float(DAY_TO_S)
                inv = invert_stehfest_vec(Fvec, t_sec, N_st) if meth=="Stehfest" else invert_gaver_euler_vec(Fvec, t_sec, M=M_gv, P=P_gv)
                pwD.append(float(inv[0]))
            pwD = np.array(pwD, float)
            dpwD = np.gradient(pwD, np.log(np.clip(tD, 1e-12, 1e12)))
        else:
            st.subheader("Kernel físico (slab no‑flujo)")
            use_dp = st.checkbox("Doble porosidad (Warren–Root)", value=False)
            if use_dp:
                omega_dp = st.number_input("ω (0..1)", 0.0, 1.0, 0.5, 0.05)
                Lambda_dp = st.number_input("Λ (transferencia)", 1e-9, 1e6, 1.0, format="%.2e")
            else:
                omega_dp, Lambda_dp = 0.5, 1.0
            par = st.session_state.params
            mu_SI = si_mu(float(par.get("mu", 0.02)))
            ct_SI = si_ct(float(par.get("phi_f", 0.45))*float(par.get("c_tf",3e-4)) + float(par.get("phi_m",0.1))*float(par.get("c_tm",2e-4)))
            k_SI = si_k(float(par.get("k_m", 0.01))*1e6)
            h_SI = si_h(float(par.get("h", 250.0)))
            L_SI = si_L(float(par.get("x_e", 250.0)))
            df_sched = st.session_state.sched
            t_sched = df_sched["t_dias"].values
            q_sched = df_sched["q_STB_d"].values
            def Fvec_phys(s: float) -> np.ndarray:
                from physics import R_slab_no_flow
                R = R_slab_no_flow(mu_SI, ct_SI, k_SI, h_SI, L_SI, float(s), use_dp=use_dp, omega=omega_dp, Lambda=Lambda_dp)
                qhat = qhat_piecewise_m3s(float(s), t_sched, q_sched)
                return np.array([R * qhat], dtype=float)
            pw_list = []
            for tt in tD:
                t_sec = float(tt) * float(DAY_TO_S)
                inv = invert_stehfest_vec(Fvec_phys, t_sec, 12)
                pw_list.append(field_p(float(inv[0])))
            pw_phys = np.array(pw_list, float)
            mu = float(par.get("mu", 0.02)); q_ref = float(par.get("q_ref", 100.0)); k = max(float(par.get("k_m", 0.01)), 1e-12); h = max(float(par.get("h", 250.0)), 1e-12)
            pwD = (pw_phys * (k * h)) / (141.2 * max(mu,1e-12) * max(q_ref,1e-12))
            dpwD = np.gradient(pwD, np.log(np.clip(tD, 1e-12, 1e12)))

        pD = np.asarray(pwD, float)
        wc = st.radio("Control de pozo", ("Caudal constante (qD=1)", "Presión constante (q = q_ref/p_wD)"), index=0)
        qD = np.ones_like(tD) if wc.startswith("Caudal") else (1.0 / np.clip(pwD, 1e-12, 1e12))
        st.session_state.update({"tD": tD, "pD": pD, "qD": qD, "pwD": pwD, "dpwD": dpwD})
        st.success("Soluciones D(t) calculadas.")

    elif step == steps[4]:
        st.header("5) Resultados y gráficos")
        req = all(k in st.session_state for k in ["tD","pD","qD","params"])
        if not req:
            st.warning("Calculá D(t) en el Paso 4.")
            st.stop()
        tD = np.asarray(st.session_state["tD"], float)
        pD = np.asarray(st.session_state["pD"], float)
        qD = np.asarray(st.session_state["qD"], float)
        mu = float(st.session_state.params.get("mu", 0.02))
        k = _clamp_pos(st.session_state.params.get("k_f", 1000.0))
        h = _clamp_pos(st.session_state.params.get("h", 250.0))
        q_ref = float(st.session_state.params.get("q_ref", 100.0))
        p_res = float(st.session_state.params.get("p_res", 3000.0))
        dP_real = p_res_from_adim(pD, mu, k, h, q_ref)
        Q_real = q_from_adim(qD, q_ref)
        Pwf = p_res - dP_real
        mask = (tD > 0) & np.isfinite(dP_real) & np.isfinite(Q_real)
        tD, dP_real, Q_real, Pwf = tD[mask], dP_real[mask], Q_real[mask], Pwf[mask]
        pD = pD[mask]
        st.subheader("Gráficos principales")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.loglog(tD, dP_real, "o-", label="ΔP [psi]", color="navy")
        ax1.set_xlabel("t_D"); ax1.set_ylabel("ΔP [psi]"); ax1.grid(True, which="both", ls="--", alpha=0.6); ax1.legend(); st.pyplot(fig1)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(tD, Pwf, "r-", label="Pwf [psi]"); ax2.set_xlabel("t_D"); ax2.set_ylabel("Pwf [psi]"); ax2.grid(True, ls="--", alpha=0.6); ax2.legend(); st.pyplot(fig2)
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.plot(tD, Q_real, "g-", label="Q [STB/d]"); ax3.set_xlabel("t_D"); ax3.set_ylabel("Q [STB/d]"); ax3.grid(True, ls="--", alpha=0.6); ax3.legend(); st.pyplot(fig3)
        dpdlogt = np.gradient(np.log(np.clip(pD, 1e-12, 1e12)), np.log(tD))
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.loglog(tD, pD, "b-", label="p_D"); ax4.loglog(tD, np.abs(dpdlogt), "k--", label="|dp_D/d ln t_D|")
        ax4.set_xlabel("t_D"); ax4.set_ylabel("p_D y derivada"); ax4.grid(True, which="both", ls="--", alpha=0.6); ax4.legend(); st.pyplot(fig4)
        st.subheader("Diagnóstico")
        if np.any(~np.isfinite(dP_real)) or np.any(~np.isfinite(Q_real)):
            st.warning("Resultados no válidos (NaN/Inf). Revisá estabilidad.")
        else:
            st.success("Cálculos estables.")
        df_res = pd.DataFrame({"t_D":tD,"p_D":pD,"ΔP_psi":dP_real,"Pwf_psi":Pwf,"Q_STB_d":Q_real})
        st.download_button("Descargar resultados (CSV)", df_res.to_csv(index=False).encode("utf-8"), "resultados.csv", "text/csv")


if __name__ == "__main__":
    app()

