# ─────────────────────────────────────────────────────────────────────────────
# SPE-215031-PA — PARTE B: R(s*), Stehfest, Pwf(t), ΔP(t), estilos/log
# Esta parte la importa la PARTE A y pinta la pestaña 4.
# ─────────────────────────────────────────────────────────────────────────────
import math
import numpy as np
import plotly.graph_objects as go

# ── Stehfest ─────────────────────────────────────────────────────────────────
def stehfest_weights(N:int)->np.ndarray:
    assert N%2==0 and N>0
    V=np.zeros(N+1)
    for k in range(1,N+1):
        s=0.0; jmin=(k+1)//2; jmax=min(k,N//2)
        for j in range(jmin,jmax+1):
            num = j**(N//2)*math.factorial(2*j)
            den = math.factorial(N//2 - j)*math.factorial(j)*math.factorial(j-1)*math.factorial(k-j)*math.factorial(2*j-k)
            s += num/den
        V[k]=s*((-1)**(k+N//2))
    return V[1:]

def invert_stehfest_vec(Fvec, t:float, N:int)->np.ndarray:
    if t<=0: return np.nan
    V=stehfest_weights(N)
    s_nodes = (np.arange(1,N+1)*math.log(2.0))/max(t,1e-30)
    vals=[np.asarray(Fvec(s), float) for s in s_nodes]
    vals=np.stack(vals, axis=0)
    return (math.log(2.0)/t) * (V[:,None]*vals).sum(axis=0)

# ── Pestaña 4: matriz & resultados ──────────────────────────────────────────
def render_results_tab(st, ss, tab,
                       build_geometry_numbers, distances_lateral_from_spacings,
                       R_self, R_cross_lateral, q_hat_piecewise_s,
                       si_mu, si_k, si_ct, si_h, si_L, field_p,
                       DAY_TO_S, STB_TO_M3):

    app = ss.app
    phys = ss.phys
    with tab:
        st.markdown("### Matriz **R(s\*)**, vector **q̂(s\*)** y resultados")
        c0,c1,c2 = st.columns([1,1,1.2])
        with c0:
            t_star_day = st.number_input("t* [day] (para s*)", 1e-6, 1e6, 10.0, format="%.6f", key="t_star")
        with c1:
            Nst = st.slider("N (Stehfest)", 8, 20, 12, 2, help="Orden de Stehfest (par).")
        with c2:
            mode_log = st.checkbox("Graficar ejes t en log", True)

        # s* y constantes SI
        s_star = (6*math.log(2.0))/(t_star_day*DAY_TO_S)  # por defecto k=6 para mostrar algo intermedio
        mu  = si_mu(phys["mu_cp"]); kI=si_k(phys["kI_nD"]); kO=si_k(phys["kO_nD"])
        ct  = si_ct(phys["ct_invpsi"]); h=si_h(phys["h_ft"])

        # Geometría ↓
        nums = build_geometry_numbers(app.spacing_g_ft, app.Lx_I_i_ft, app.two_xO_i_ft)
        D_ft = distances_lateral_from_spacings(app.spacing_g_ft)
        D = si_L(D_ft)

        # Armar R(s*)
        Nw = app.N
        R = np.zeros((Nw,Nw), float)
        Ls = si_L(np.mean(app.Lx_I_i_ft))     # usamos un Lx_SRV representativo (opción simple p/ejemplo)
        # Para ORV_end usamos el semi-ancho del pozo (tomamos de la izquierda; equivalente, podrías promediar)
        Lo = si_L(np.mean(nums["Lx_O_end_i_ft"]) if nums["Lx_O_end_i_ft"] else 0.0)

        for i in range(Nw):
            R[i,i] = R_self(mu, ct, kI, kO, h, Ls, Lo, s_star,
                            ss.phys["dp_on_I"], ss.phys["omega_I"], ss.phys["Lambda_I"],
                            ss.phys["dp_on_O"], ss.phys["omega_O"], ss.phys["Lambda_O"])
        for i in range(Nw):
            for j in range(Nw):
                if i!=j:
                    R[i,j] = R_cross_lateral(mu, ct, kO, h, D[i,j], s_star)

        # Unidades a Field (psi·day/STB)
        Rfield = (R/6894.757293168) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)

        # q̂(s*)
        qhat = np.array([q_hat_piecewise_s(w.sched, s_star) for w in app.wells[:Nw]], float)

        # Mostrar R y q̂
        st.markdown("**Matriz** $\\mathbf R(s^*)$ [psi·day/STB] y **vector** $\\hat q(s^*)$ [STB]:")
        figR = go.Figure(data=go.Heatmap(z=Rfield, x=[f"q̂{j+1}" for j in range(Nw)],
                                         y=[f"p̂{j+1}" for j in range(Nw)], colorscale="Blues"))
        figR.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(figR, use_container_width=True)
        st.code("p̂(s*) = R(s*) · q̂(s*)   (en unidades field: psi·day / STB)")

        # Serie temporal (Pwf y ΔP)
        st.markdown("#### Resultados en el tiempo (inversión de Laplace)")
        c3,c4,c5 = st.columns([1,1,1])
        with c3:
            tmin_d = st.number_input("t_min [day]", 1e-6, 1e2, 1e-3, format="%.6f")
        with c4:
            tmax_d = st.number_input("t_max [day]", 1e-4, 1e6, 1e3, format="%.4f")
        with c5:
            npts   = st.number_input("n_pts", 16, 2000, 220, 1)

        times_s = np.logspace(np.log10(tmin_d), np.log10(tmax_d), int(npts))*DAY_TO_S

        # Definir Fvec que devuelve p̂(s) [psi·day] para todo el array (tal como en la parte A)
        def Fvec(s):
            # replicamos la construcción con este s puntual
            R = np.zeros((Nw,Nw), float)
            for i in range(Nw):
                R[i,i] = R_self(mu, ct, kI, kO, h, Ls, Lo, s,
                                ss.phys["dp_on_I"], ss.phys["omega_I"], ss.phys["Lambda_I"],
                                ss.phys["dp_on_O"], ss.phys["omega_O"], ss.phys["Lambda_O"])
            for i in range(Nw):
                for j in range(Nw):
                    if i!=j:
                        R[i,j] = R_cross_lateral(mu, ct, kO, h, D[i,j], s)
            qhat_loc = np.array([q_hat_piecewise_s(w.sched, s) for w in app.wells[:Nw]], float)
            p_hat_SI = R.dot(qhat_loc * STB_TO_M3 / DAY_TO_S)     # [Pa·s]
            return (p_hat_SI/6894.757293168) * DAY_TO_S            # [psi·day]

        P = np.zeros((len(times_s), Nw))
        p_res = ss.phys["p0_psi"]
        for i,ti in enumerate(times_s):
            pwf_hat = invert_stehfest_vec(Fvec, ti, Nst)   # [psi]
            P[i,:] = p_res - pwf_hat

        # Plot Pwf por pozo
        fig = go.Figure()
        for j in range(Nw):
            fig.add_scatter(x=times_s/86400.0, y=P[:,j], mode="lines", name=f"w{j+1}: Pwf [psi]")
        fig.update_xaxes(title="t [day]", type="log" if mode_log else "linear")
        fig.update_yaxes(title="Pwf [psi]")
        fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # ΔP entre el 1 y el resto (para ver interferencia)
        if Nw>=2:
            fig2=go.Figure()
            for j in range(1,Nw):
                fig2.add_scatter(x=times_s/86400.0, y=P[:,0]-P[:,j], mode="lines", name=f"ΔP[w1−w{j+1}] [psi]")
            fig2.update_xaxes(title="t [day]", type="log" if mode_log else "linear")
            fig2.update_yaxes(title="ΔP [psi]")
            fig2.update_layout(height=320, legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)

        st.info("Con **gaps pequeños** y **k_O alto** la **interferencia** se ve fuerte (ΔP grandes). "
                "El ejemplo 'Interferencia FUERTE' está preconfigurado para evidenciarlo.")