# SOLO SPE-215031-PA — Interferencia entre 2 pozos paralelos (desde cero)
# v2: mantiene schedules de usuario, glosario+siglas, instrucciones por pestaña,
#     vista en planta con colores diferenciados, álgebra con Green promediada,
#     resultados con estilo configurable (log/linear, marcadores, dashes, símbolos).

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# ===================== UI BASE =====================
st.set_page_config(page_title="SPE-215031-PA — Pozos paralelos (v2)", layout="wide")
st.title("SPE-215031-PA — Interferencia de 2 pozos paralelos (v2)")

# ===================== GLOSARIO + SIGLAS =====================
@dataclass
class Sym:
    latex: str
    unit: str
    desc: str

def _h(k): s=SYMS[k]; return f"${s.latex}$ · [{s.unit}] — {s.desc}"

SYMS: Dict[str, Sym] = {
    "mu":  Sym(r"\mu", "cP", "Viscosidad del fluido"),
    "k":   Sym(r"k", "nD", "Permeabilidad efectiva"),
    "ct":  Sym(r"c_t", "1/psi", "Compresibilidad total"),
    "h":   Sym(r"h", "ft", "Espesor de la capa"),
    "p0":  Sym(r"p_{\mathrm{res}}", "psi", "Presión inicial del yacimiento"),
    # Geometría
    "nF":  Sym(r"n_F", "–",  "Número de fracturas por pozo"),
    "xF":  Sym(r"x_F", "ft", "Semi-espaciamiento entre fracturas (spacing = 2·x_F)"),
    "LxSRV": Sym(r"L_{x,\mathrm{SRV}}", "ft", "Semi-longitud SRV a lo largo del pozo"),
    "LxORVend": Sym(r"L_{x,\mathrm{ORV,end}}", "ft", "Extensión ORV a extremos"),
    "D12": Sym(r"D_{12}", "ft", "Separación entre pozos (paralelos)"),
    # Kernels
    "lam": Sym(r"\lambda", "1/ft", "Parámetro difusivo en Laplace (escala espacial)"),
    "qhat": Sym(r"\hat q(s)", "STB", "Transformada de Laplace del caudal por tramos"),
}

ABBR = {
    "SRV":"Stimulated Reservoir Volume (volumen estimulado)",
    "ORV":"Outer Reservoir Volume (volumen externo)",
    "BHP":"Bottom-Hole Pressure (presión de fondo de pozo)",
    "Pwf":"Presión en el fondo de pozo (well flowing pressure)",
    "GSI":"Green (función de Green) — solución fundamental promediada sobre geometrías",
}

# ===================== CONVERSIONES FIELD↔SI =====================
PSI_TO_PA = 6894.757293168
FT_TO_M   = 0.3048
CP_TO_PAS = 1e-3
ND_TO_M2  = 9.869233e-22
DAY_TO_S  = 86400.0
STB_TO_M3 = 0.158987294928

def si_mu(mu_cp):  return mu_cp*CP_TO_PAS
def si_k(k_nD):    return k_nD*ND_TO_M2
def si_ct(ct_invpsi): return ct_invpsi/PSI_TO_PA
def si_h(h_ft):    return h_ft*FT_TO_M
def si_L(L_ft):    return L_ft*FT_TO_M
def field_p(p_pa): return p_pa/PSI_TO_PA

# ===================== ESTABILIDAD NUMÉRICA =====================
def tanh_stable(x):
    x=np.asarray(x,float); out=np.empty_like(x)
    big=x>20.0; sml=x<-20.0; mid=~(big|sml)
    out[big]=1.0; out[sml]=-1.0; out[mid]=np.tanh(x[mid]); return out
def coth_stable(x):
    x=np.asarray(x,float); ax=np.abs(x); out=np.empty_like(x)
    tiny=ax<1e-8; out[tiny]=1.0/x[tiny]+x[tiny]/3.0; out[~tiny]=1.0/tanh_stable(x[~tiny]); return out
def exp_clamped(z, lim=700.0): return np.exp(np.clip(z, -lim, lim))

# ===================== INVERSION DE LAPLACE =====================
if "fs_calls_stehfest" not in st.session_state: st.session_state.fs_calls_stehfest=0
if "fs_calls_gaver" not in st.session_state:    st.session_state.fs_calls_gaver=0
def _bump_steh(n=1): st.session_state.fs_calls_stehfest += int(n)
def _bump_gav(n=1):  st.session_state.fs_calls_gaver    += int(n)

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
    vals=[np.asarray(Fvec(s), float) for s in s_nodes if not _bump_steh() ]
    vals=np.stack(vals, axis=0)
    return (math.log(2.0)/t) * (V[:,None]*vals).sum(axis=0)

def invert_gaver_euler_vec(Fvec, t:float, M:int=18, P:int=8)->np.ndarray:
    ln2=math.log(2.0)
    S=[]
    for n in range(1, M+P+1):
        acc=None
        for k in range(1, n+1):
            s = k*ln2/max(t,1e-30); _bump_gav()
            term = ((-1)**k)*math.comb(n,k)*np.asarray(Fvec(s),float)
            acc = term if acc is None else acc+term
        S.append(acc)
    S=np.stack(S, axis=0)
    E = sum(math.comb(P,p)*S[(M-1)+p] for p in range(0,P+1))/(2.0**P)
    return (-ln2/max(t,1e-30)) * E

# ===================== NUCLEO FISICO (LAPLACE) =====================
def _lambda(mu, ct, k, s):  return math.sqrt(max(s,1e-40)*max(mu*ct,1e-40)/max(k,1e-40))

def R_slab_no_flow(mu, ct, k, h, L, s):
    lam = _lambda(mu, ct, k, s); x = lam*max(L,1e-12)
    return (mu/(k*max(h,1e-12))) * (coth_stable(x)/max(lam,1e-30))

def R_semi_inf(mu, ct, k, h, s):
    lam = _lambda(mu, ct, k, s); return (mu/(k*max(h,1e-12))) * (1.0/max(lam,1e-30))

def R_self(mu, ct, k_srv, k_orv, h, Lx_srv, Lx_orv_end, s):
    R_srv = R_slab_no_flow(mu, ct, k_srv, h, Lx_srv, s)
    R_orv = R_slab_no_flow(mu, ct, k_orv, h, Lx_orv_end, s) if Lx_orv_end>0 else R_semi_inf(mu, ct, k_orv, h, s)
    return R_srv + R_orv

def R_cross(mu, ct, k_orv, h, Dij, s):
    lam = _lambda(mu, ct, k_orv, s)
    return (mu/(k_orv*max(h,1e-12))) * float(exp_clamped(-lam*max(Dij,0.0))) / max(lam,1e-30)

# ===================== SCHEDULE (q̂ POR TRAMOS) =====================
@dataclass
class WellSched:
    t_days: List[float]
    q_stbd: List[float]

def ensure_schedule(ws: WellSched)->WellSched:
    if not ws.t_days: ws.t_days=[0.0]
    if not ws.q_stbd: ws.q_stbd=[0.0]
    pairs=sorted(zip(ws.t_days, ws.q_stbd), key=lambda z:z[0])
    t=[pairs[0][0]]; q=[pairs[0][1]]
    for ti,qi in pairs[1:]:
        if ti!=t[-1]: t.append(ti); q.append(qi)
        else: q[-1]=qi
    return WellSched(t_days=t, q_stbd=q)

def q_hat_piecewise_s(ws: WellSched, s: float) -> float:
    acc=0.0; last_q=0.0
    for tk_day, qk in zip(ws.t_days, ws.q_stbd):
        tk = tk_day*DAY_TO_S
        acc += (qk-last_q)*math.exp(-s*tk)
        last_q=qk
    return acc/max(s,1e-30)  # [STB]

def schedule_step_xy(ws: WellSched, tmin: float, tmax: float, n: int=200)->Tuple[np.ndarray,np.ndarray]:
    # devuelve curva escalón q(t) en [tmin,tmax] días
    t=np.linspace(tmin, tmax, n); q=np.zeros_like(t)
    ts=np.array(ws.t_days); qs=np.array(ws.q_stbd)
    for i,ti in enumerate(t):
        idx=np.searchsorted(ts, ti, side="right")-1
        idx=max(idx,0); q[i]=qs[idx]
    return t,q

# ===================== GEOMETRIA: PARALELO FORZADO (2 POZOS) =====================
@dataclass
class ParallelGeom:
    nF: int        # fracturas por pozo
    xF_ft: float   # semi-espaciamiento entre fracturas (spacing = 2*xF)
    Lx_orv_end_ft: float
    D12_ft: float  # spacing entre pozos

def frac_centers_x(nF: int, xF: float)->np.ndarray:
    # posiciones x_k simétricas: k=1..nF → (k-(nF+1)/2)*2xF
    ks = np.arange(1, nF+1, dtype=float)
    return (ks - (nF+1)/2.0) * (2.0*xF)

def plan_plot(geom: ParallelGeom, Lx_srv_ft: float, sel_w:int=None, sel_k:int=None):
    # wells at y = -D/2 and +D/2
    y1, y2 = -geom.D12_ft/2.0, +geom.D12_ft/2.0
    xk = frac_centers_x(geom.nF, geom.xF_ft)
    L = Lx_srv_ft
    # style (colores distintos por pozo)
    w1_col = "#1e88e5"; w2_col = "#8e24aa"
    w1_bg = "#e3f2fd";  w2_bg = "#f3e5f5"
    orv_col = "#90caf9"
    fig, ax = plt.subplots(figsize=(11,6))
    # ORV strip
    x_left = -(L + geom.Lx_orv_end_ft); x_right = +(L + geom.Lx_orv_end_ft)
    ax.add_patch(plt.Rectangle((x_left, y1-120), x_right-x_left, geom.D12_ft+240, color=orv_col, alpha=0.25, lw=0, zorder=0))
    # Background bands per well
    ax.add_patch(plt.Rectangle((x_left, y1-40), x_right-x_left, 80, color=w1_bg, alpha=0.8, lw=0, zorder=1))
    ax.add_patch(plt.Rectangle((x_left, y2-40), x_right-x_left, 80, color=w2_bg, alpha=0.8, lw=0, zorder=1))
    # Wells (bars)
    ax.plot([-L, L], [y1, y1], color=w1_col, lw=8, solid_capstyle="round", zorder=2)
    ax.plot([-L, L], [y2, y2], color=w2_col, lw=8, solid_capstyle="round", zorder=2)
    ax.text(0, y1-55, "Well 1 (i=1)", ha="center", va="top", fontsize=11, color=w1_col, zorder=3)
    ax.text(0, y2+55, "Well 2 (i=2)", ha="center", va="bottom", fontsize=11, color=w2_col, zorder=3)
    # Fractures (vertical ticks)
    for idx, x in enumerate(xk, start=1):
        ax.plot([x, x], [y1-18, y1+18], color="black", lw=3, zorder=3)
        ax.text(x, y1+26, f"k={idx}", ha="center", va="bottom", fontsize=9, zorder=3)
        ax.plot([x, x], [y2-18, y2+18], color="black", lw=3, zorder=3)
        ax.text(x, y2-26, f"k={idx}", ha="center", va="top", fontsize=9, zorder=3)
    # Highlight selection
    if sel_w in (1,2) and sel_k is not None and 1<=sel_k<=geom.nF:
        y_sel = y1 if sel_w==1 else y2
        x_sel = xk[sel_k-1]
        ax.plot([x_sel, x_sel], [y_sel-26, y_sel+26], color="#000", lw=5, zorder=4)
        ax.scatter([x_sel],[y_sel], s=140, facecolor="yellow", edgecolor="black", zorder=5)
        ax.text(x_sel+8, y_sel+8, f"(i={sel_w}, k={sel_k})", fontsize=10, color="black", zorder=6)
    # Dimensions
    def dimline(x0,y0,x1,y1,txt,off=12):
        ax.annotate("", xy=(x0,y0), xytext=(x1,y1), arrowprops=dict(arrowstyle="<->", lw=1.6, color="#333"))
        xm,ym=(x0+x1)/2.0,(y0+y1)/2.0; ax.text(xm,ym+off,txt,ha="center",va="bottom",fontsize=10,color="#333")
    dimline(-L, y1-70, +L, y1-70, f"2·Lx_SRV = {2*L:.1f} ft")
    if len(xk)>=2:
        dimline(xk[0], y2+70, xk[1], y2+70, f"2·x_F = {2*geom.xF_ft:.1f} ft")
    xr = x_right + 140
    ax.annotate("", xy=(xr, y1), xytext=(xr, y2), arrowprops=dict(arrowstyle="<->", lw=1.6, color="#333"))
    ax.text(xr+10, 0, f"D12 = {geom.D12_ft:.1f} ft", rotation=90, va="center", ha="left", fontsize=10, color="#333")
    dimline(-L-geom.Lx_orv_end_ft, y1-110, -L, y1-110, f"Lx_ORV,end = {geom.Lx_orv_end_ft:.0f} ft")
    # Aesthetics
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_left-200, x_right+200)
    ax.set_ylim(y1-180, y2+180)
    ax.set_xlabel("x (ft) — vista en planta"); ax.set_ylabel("y (ft)")
    ax.set_title("Pozos paralelos (2) con n_F fracturas — resaltar i-ésimo/k-ésimo elemento")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ===================== ESTADO INICIAL (schedules) =====================
if "sched1" not in st.session_state:
    st.session_state.sched1 = WellSched([0.0, 10.0, 30.0], [800.0, 600.0, 0.0])
if "sched2" not in st.session_state:
    st.session_state.sched2 = WellSched([0.0, 10.0, 30.0], [0.0,  400.0, 0.0])

# ===================== TABS =====================
tab_intro, tab_props, tab_sched, tab_geom, tab_alg, tab_results, tab_tables, tab_gloss = st.tabs([
    "0) Intro",
    "1) Parámetros fluido/roca",
    "2) Pozos & Schedules",
    "3) Geometría (planta)",
    "4) Álgebra (SPE-215031-PA)",
    "5) Resultados",
    "6) Tablas i/k/n-ésimas",
    "Glosario + Siglas",
])

# ===================== 0) INTRO =====================
with tab_intro:
    st.markdown("""
**¿Qué hay en cada pestaña?**
- **1) Parámetros fluido/roca:** ingresá μ, k_SRV, k_ORV, c_t, h y p_res (Field Units).
- **2) Pozos & Schedules:** cargá los escalones de caudal de **cada pozo** (t_k en días y q_k en STB/D). Podés previsualizar q(t).
- **3) Geometría:** definí n_F, x_F, Lx_SRV (opcional: vincular Lx_SRV=(n_F−1)·x_F), D12 y Lx_ORV,end. Resaltá el elemento i/k.
- **4) Álgebra:** se muestra la formulación en Laplace (Green promediada), y el cálculo de R(s*)·q̂(s*) con **tus** entradas.
- **5) Resultados:** Pwf(t), ΔP_res(t)=p_res−Pwf(t), y ΔP₁₂(t) entre pozos. Estilo editable (log/linear, marcadores, trazos).
- **6) Tablas:** coordenadas k-ésimas, métricas n-ésimas y distancias i-ésimas.
- **Glosario + Siglas:** referencia clara de símbolos, unidades y abreviaturas.
""")

# ===================== 1) PROPIEDADES =====================
with tab_props:
    st.markdown("**Qué hacer acá:** ingresá propiedades del fluido/roca en unidades de campo (Field).")
    c1,c2,c3 = st.columns(3)
    with c1:
        mu_cp   = st.number_input(_h("mu"), 0.01, 50.0, 1.0, 0.01)
        k_srv_nD= st.number_input("k_SRV [nD] — Permeabilidad SRV", 0.1, 1e6, 800.0, 0.1)
    with c2:
        k_orv_nD= st.number_input("k_ORV [nD] — Permeabilidad ORV", 0.1, 1e6, 150.0, 0.1)
        ct_invpsi = st.number_input(_h("ct"), 1e-7, 1e-2, 1.0e-6, format="%.2e")
    with c3:
        h_ft     = st.number_input(_h("h"), 1.0, 300.0, 65.6, 0.1)
        p0_psi   = st.number_input(_h("p0"), 100.0, 10000.0, 4350.0, 1.0)

# ===================== 2) POZOS & SCHEDULES =====================
with tab_sched:
    st.markdown("""
**Cómo cargar el schedule:**  
- Ingresá **listas separadas por comas** para tiempos `t_k [day]` y caudales `q_k [STB/D]` en cada pozo.  
- Deben tener el **mismo tamaño**; la app ordena por tiempo y consolida tiempos repetidos.
- Usá los botones **Actualizar** para aplicar cambios y **Previsualizar** para ver q(t).
""")
    def edit_sched(title: str, key_prefix: str, ws: WellSched)->WellSched:
        with st.expander(title, expanded=True):
            c1,c2 = st.columns(2)
            with c1:
                t_line = st.text_input("t_k [day]", ", ".join(str(t) for t in ws.t_days), key=key_prefix+"_t")
            with c2:
                q_line = st.text_input("q_k [STB/D]", ", ".join(str(q) for q in ws.q_stbd), key=key_prefix+"_q")
            c3,c4 = st.columns(2)
            updated = ws
            with c3:
                if st.button("Actualizar", key=key_prefix+"_upd"):
                    try:
                        t_vals=[float(v.strip()) for v in t_line.split(",") if v.strip()!=""]
                        q_vals=[float(v.strip()) for v in q_line.split(",") if v.strip()!=""]
                        updated = ensure_schedule(WellSched(t_vals, q_vals))
                        st.success("Schedule actualizado.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            with c4:
                if st.button("Previsualizar", key=key_prefix+"_plot"):
                    tmin = 0.0; tmax = max(updated.t_days[-1], 1.0)
                    tt,qq = schedule_step_xy(updated, tmin, tmax, n=400)
                    fig = go.Figure()
                    fig.add_scatter(x=tt, y=qq, mode="lines", name="q(t) [STB/D]")
                    fig.update_xaxes(title="t [day]"); fig.update_yaxes(title="q [STB/D]")
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)
            return updated

    st.session_state.sched1 = edit_sched("Pozo 1 — Schedule", "w1", st.session_state.sched1)
    st.session_state.sched2 = edit_sched("Pozo 2 — Schedule", "w2", st.session_state.sched2)

# ===================== 3) GEOMETRIA (PLANTA) =====================
with tab_geom:
    st.markdown("**Qué hacer acá:** definí la geometría y resaltá elementos i/k.")
    c1,c2,c3 = st.columns(3)
    with c1:
        nF = st.number_input(_h("nF"), 1, 50, 3, 1)
        xF_ft = st.number_input(_h("xF"), 1.0, 1000.0, 50.0, 1.0)
    with c2:
        link_srv = st.checkbox("Vincular Lx_SRV = (n_F−1)·x_F", True)
        Lx_srv_ft = (nF-1)*xF_ft if link_srv else st.number_input(_h("LxSRV"), 1.0, 50000.0, 100.0, 1.0)
    with c3:
        D12_ft = st.number_input(_h("D12"), 1.0, 100000.0, 660.0, 1.0)
        Lx_orv_end_ft = st.number_input(_h("LxORVend"), 0.0, 200000.0, 2000.0, 10.0)

    geom = ParallelGeom(nF=nF, xF_ft=xF_ft, Lx_orv_end_ft=Lx_orv_end_ft, D12_ft=D12_ft)

    c4,c5 = st.columns(2)
    with c4:
        sel_w = st.selectbox("Elegí pozo i (para resaltar)", [None,1,2], index=0)
    with c5:
        sel_k = st.selectbox("Elegí fractura k (para resaltar)", [None]+list(range(1, nF+1)), index=0)
    plan_plot(geom, Lx_srv_ft, sel_w, sel_k)

# ===================== 4) ALGEBRA (SPE-215031-PA) =====================
with tab_alg:
    st.markdown("""**Qué hay acá:** derivamos y evaluamos **R(s)\u00b7q̂(s)** con tus propiedades, geometría y **schedules**.
- **q̂(s)** por tramos, **λ(s)**, **R_slab** (SRV+ORV end) y **R_cross**(D12) (Green promediada).
- Elegís un **s*** (vía k de Stehfest y t*), y vemos los valores numéricos.
""")
    # 1) q̂(s) por tramos (USUARIO)
    st.markdown("**1) Transformada del caudal por tramos (por pozo)**")
    st.latex(r"\hat q_j(s)=\frac{1}{s}\sum_{k}\big(q_k-q_{k-1}\big)\,e^{-s\,t_{k-1}}")
    c1,c2 = st.columns(2)
    with c1:
        t_star_day = st.number_input("t* [day]", 1e-6, 1e6, 10.0, format="%.6f")
    with c2:
        k_idx = st.slider("k (nodo Stehfest)", 1, 20, 6, 1)
    s_star = (k_idx*math.log(2.0))/(t_star_day*DAY_TO_S)
    st.latex(rf"s^*=\dfrac{{{k_idx}\,\ln 2}}{{{t_star_day}\ \mathrm{{day}}}}={s_star:.3e}\ \mathrm{{s^{{-1}}}}")

    ws1 = ensure_schedule(st.session_state.sched1)
    ws2 = ensure_schedule(st.session_state.sched2)
    qhat1 = q_hat_piecewise_s(ws1, s_star)
    qhat2 = q_hat_piecewise_s(ws2, s_star)
    st.markdown("**Vector** $\\hat{\\mathbf q}(s^*)$ [STB]:")
    st.latex(rf"\hat{{\mathbf q}}(s^*)=\begin{{bmatrix}}{qhat1:.3e}\\ {qhat2:.3e}\end{{bmatrix}}\ \mathrm{{STB}}")

    # 2) λ(s), R(s): propios y cruzados
    st.markdown("**2) Parámetro difusivo y funciones de Green promediada**")
    st.latex(r"\lambda=\sqrt{\frac{\mu\,c_t}{k}\,s}")
    st.latex(r"\mathcal R_{\text{slab}}=\frac{\mu}{k\,h}\,\frac{\coth(\lambda L)}{\lambda},\quad"
             r"\mathcal R_{\text{semiinf}}=\frac{\mu}{k\,h}\,\frac{1}{\lambda}")
    st.latex(r"""\mathcal R_{11}=\mathcal R_{22}=\frac{\mu}{h}\left[
\frac{1}{k_{\mathrm{SRV}}}\,\frac{\coth(\lambda_{\mathrm{SRV}}\,L_{x,\mathrm{SRV}})}{\lambda_{\mathrm{SRV}}}+
\frac{1}{k_{\mathrm{ORV}}}\,\frac{\coth(\lambda_{\mathrm{ORV}}\,L_{x,\mathrm{ORV,end}})}{\lambda_{\mathrm{ORV}}}\right]""")
    st.latex(r"""\mathcal R_{12}=\mathcal R_{21}=\frac{\mu}{k_{\mathrm{ORV}}\,h}\,
\frac{e^{-\lambda_{\mathrm{ORV}}\,D_{12}}}{\lambda_{\mathrm{ORV}}}""")

    # Sustitución numérica
    mu_SI, ct_SI = si_mu(mu_cp), si_ct(ct_invpsi)
    kSRV_SI, kORV_SI = si_k(k_srv_nD), si_k(k_orv_nD)
    h_SI = si_h(h_ft); Ls_SI = si_L(Lx_srv_ft); Lo_SI = si_L(Lx_orv_end_ft); D12_SI = si_L(D12_ft)
    lam_SRV = _lambda(mu_SI, ct_SI, kSRV_SI, s_star)
    lam_ORV = _lambda(mu_SI, ct_SI, kORV_SI, s_star)
    lam_SRV_ft = lam_SRV*FT_TO_M; lam_ORV_ft = lam_ORV*FT_TO_M

    # R(s*) en Field units [psi·day/STB]
    R11_SI = R_self(mu_SI, ct_SI, kSRV_SI, kORV_SI, h_SI, Ls_SI, Lo_SI, s_star)
    R12_SI = R_cross(mu_SI, ct_SI, kORV_SI, h_SI, D12_SI, s_star)
    conv = (1/PSI_TO_PA) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)
    R11 = R11_SI*conv; R12=R12_SI*conv
    R = np.array([[R11, R12],[R12, R11]], float)

    st.markdown("**Valores usados (este s*)**")
    st.write(f"λ_SRV = {lam_SRV_ft:.3e} 1/ft, λ_ORV = {lam_ORV_ft:.3e} 1/ft")

    st.markdown("**Sistema**  $\\hat{\\mathbf p}(s)=\\mathbf R(s)\\,\\hat{\\mathbf q}(s)$")
    figR = go.Figure(data=go.Heatmap(z=R, x=["q̂₁","q̂₂"], y=["p̂₁","p̂₂"], colorscale="Blues"))
    figR.update_layout(height=300)
    st.plotly_chart(figR, use_container_width=True)

    st.markdown("**Ecuaciones por renglón (con valores) [psi·day]**")
    st.latex(rf"\hat p_1={R11:.3e}\,\hat q_1 + {R12:.3e}\,\hat q_2")
    st.latex(rf"\hat p_2={R12:.3e}\,\hat q_1 + {R11:.3e}\,\hat q_2")

    p1 = R11*qhat1 + R12*qhat2
    p2 = R12*qhat1 + R11*qhat2
    st.markdown("**Vector** $\\hat{\\mathbf p}(s^*)$ [psi·day]:")
    st.latex(rf"\hat{{\mathbf p}}(s^*)=\begin{{bmatrix}}{p1:.3e}\\ {p2:.3e}\end{{bmatrix}}\ \mathrm{{psi\cdot day}}")

# ===================== 5) RESULTADOS =====================
with tab_results:
    st.markdown("""
**Qué ver acá:** Pwf(t) para cada pozo, **ΔP_res(t)=p_res−Pwf(t)** (por pozo), y **ΔP₁₂(t)** entre pozos.  
Usá **Opciones de estilo** para log/linear, marcadores, símbolos y trazos.
""")
    # Estilo general
    with st.expander("Opciones de estilo", expanded=False):
        log_x = st.checkbox("Eje X logarítmico", True)
        log_y_P = st.checkbox("Y log (presiones y ΔP)", False)
        log_y_q = st.checkbox("Y log (caudales)", False)
        line_dash = st.selectbox("Trazo", ["solid","dash","dot","dashdot"], index=0)
        show_markers = st.checkbox("Mostrar marcadores", True)
        marker_symbol = st.selectbox("Símbolo", ["circle","square","triangle-up","diamond","cross","x","pentagon","star"], index=0)
        marker_size = st.slider("Tamaño marcador", 4, 16, 7, 1)
        line_width = st.slider("Espesor de línea", 1, 6, 2, 1)

    # Inversión
    c1,c2,c3 = st.columns(3)
    with c1: inv_m = st.radio("Método", ["Stehfest","Gaver-Euler"], index=0)
    with c2: Nst   = st.slider("N (Stehfest)", 8, 20, 12, 2)
    with c3:
        M_gv = st.slider("M (Gaver)", 12, 40, 18, 2)
        P_gv = st.slider("P (Euler)", 6, 14, 8, 1)

    c4,c5,c6 = st.columns(3)
    with c4: tmin_d = st.number_input("t_min [day]", 1e-6, 1e2, 1e-3, format="%.6f")
    with c5: tmax_d = st.number_input("t_max [day]", 1e-4, 1e6, 1e3, format="%.4f")
    with c6: npts   = st.number_input("n_pts", 16, 2000, 220, 1)
    times_s = np.logspace(np.log10(tmin_d), np.log10(tmax_d), int(npts))*DAY_TO_S
    ws1 = ensure_schedule(st.session_state.sched1)
    ws2 = ensure_schedule(st.session_state.sched2)

    # Fvec(s) a partir de schedules del usuario
    def Fvec(s: float)->np.ndarray:
        mu  = si_mu(mu_cp); kSRV= si_k(k_srv_nD); kORV= si_k(k_orv_nD)
        ct  = si_ct(ct_invpsi); h=si_h(h_ft); Ls=si_L(Lx_srv_ft); Lo=si_L(Lx_orv_end_ft); D12=si_L(D12_ft)
        R11_SI = R_self(mu, ct, kSRV, kORV, h, Ls, Lo, s)
        R12_SI = R_cross(mu, ct, kORV, h, D12, s)
        conv = (1/PSI_TO_PA) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)  # [psi·day / STB]
        R11 = R11_SI*conv; R12=R12_SI*conv
        qhat = np.array([q_hat_piecewise_s(ws1, s), q_hat_piecewise_s(ws2, s)], float)  # [STB]
        p_hat = np.array([R11*qhat[0] + R12*qhat[1],
                          R12*qhat[0] + R11*qhat[1]], float)  # [psi·day]
        return p_hat

    st.session_state.fs_calls_stehfest=0; st.session_state.fs_calls_gaver=0
    P = np.zeros((len(times_s), 2))  # Pwf
    for i,ti in enumerate(times_s):
        p_hat = invert_stehfest_vec(Fvec, ti, Nst) if inv_m=="Stehfest" else invert_gaver_euler_vec(Fvec, ti, M=M_gv, P=P_gv)
        P[i,:]= p0_psi - p_hat  # Pwf(t) = p_res − L^{-1}[p̂]

    # Utilidades estilo
    def line_kwargs():
        return dict(mode="lines+markers" if show_markers else "lines",
                    line=dict(width=line_width, dash=line_dash),
                    marker=dict(symbol=marker_symbol, size=marker_size))

    # (a) Pwf
    with st.expander("Curvas de presión Pwf(t)", expanded=True):
        fig=go.Figure()
        fig.add_scatter(x=times_s/DAY_TO_S, y=P[:,0], name="Well 1 — Pwf [psi]", **line_kwargs())
        fig.add_scatter(x=times_s/DAY_TO_S, y=P[:,1], name="Well 2 — Pwf [psi]", **line_kwargs())
        fig.update_xaxes(title="t [day]", type="log" if log_x else "linear")
        fig.update_yaxes(title="Pwf [psi]", type="log" if log_y_P else "linear")
        fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    # (b) ΔP_res = p_res − Pwf
    with st.expander("Curvas ΔP_res(t) = p_res − Pwf(t)", expanded=False):
        dP = p0_psi - P  # = p_hat
        fig2=go.Figure()
        fig2.add_scatter(x=times_s/DAY_TO_S, y=dP[:,0], name="Well 1 — ΔP_res [psi]", **line_kwargs())
        fig2.add_scatter(x=times_s/DAY_TO_S, y=dP[:,1], name="Well 2 — ΔP_res [psi]", **line_kwargs())
        fig2.update_xaxes(title="t [day]", type="log" if log_x else "linear")
        fig2.update_yaxes(title="ΔP_res [psi]", type="log" if log_y_P else "linear")
        fig2.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    # (c) ΔP12 = (p_res − Pwf1) − (p_res − Pwf2) = Pwf2 − Pwf1
    with st.expander("Curva ΔP₁₂(t) = (p_res − Pwf₁) − (p_res − Pwf₂)", expanded=False):
        dP12 = P[:,1] - P[:,0]
        fig3=go.Figure()
        fig3.add_scatter(x=times_s/DAY_TO_S, y=dP12, name="ΔP₁₂ [psi]", **line_kwargs())
        fig3.update_xaxes(title="t [day]", type="log" if log_x else "linear")
        fig3.update_yaxes(title="ΔP₁₂ [psi]", type="log" if log_y_P else "linear")
        fig3.update_layout(height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig3, use_container_width=True)

    # (d) Caudales q(t): desde schedules
    with st.expander("Curvas de caudal q(t) — schedules", expanded=False):
        tmin = 0.0; tmax = max(ws1.t_days[-1], ws2.t_days[-1], 1.0)
        tt1,qq1 = schedule_step_xy(ws1, tmin, tmax, 400)
        tt2,qq2 = schedule_step_xy(ws2, tmin, tmax, 400)
        figQ=go.Figure()
        figQ.add_scatter(x=tt1, y=qq1, name="Well 1 — q [STB/D]", **line_kwargs())
        figQ.add_scatter(x=tt2, y=qq2, name="Well 2 — q [STB/D]", **line_kwargs())
        figQ.update_xaxes(title="t [day]", type="log" if log_x else "linear")
        figQ.update_yaxes(title="q [STB/D]", type="log" if log_y_q else "linear")
        figQ.update_layout(height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(figQ, use_container_width=True)

    st.caption(f"Llamadas a F(s) — Stehfest: {st.session_state.fs_calls_stehfest} · Gaver(Euler): {st.session_state.fs_calls_gaver}")

# ===================== 6) TABLAS i/k/n-ésimas =====================
with tab_tables:
    st.markdown("**Qué hay acá:** listamos elementos i/k/n-ésimos de la geometría.")
    # Si no pasaste por Geometría, ponemos defaults
    try:
        geom
    except NameError:
        nF=3; xF_ft=50.0; Lx_srv_ft=100.0; Lx_orv_end_ft=2000.0; D12_ft=660.0
        geom = ParallelGeom(nF=nF, xF_ft=xF_ft, Lx_orv_end_ft=Lx_orv_end_ft, D12_ft=D12_ft)
    xk = frac_centers_x(geom.nF, geom.xF_ft); y1, y2 = -geom.D12_ft/2.0, +geom.D12_ft/2.0
    rows = []
    for k in range(1, geom.nF+1):
        rows.append(dict(elemento=f"f1.{k} (i=1,k={k})", x_ft=float(xk[k-1]), y_ft=y1))
        rows.append(dict(elemento=f"f2.{k} (i=2,k={k})", x_ft=float(xk[k-1]), y_ft=y2))
    rows.append(dict(elemento="n_F", valor=geom.nF))
    rows.append(dict(elemento="2·x_F [ft]", valor=2*geom.xF_ft))
    rows.append(dict(elemento="2·Lx_SRV [ft]", valor=2*Lx_srv_ft))
    rows.append(dict(elemento="D12 [ft]", valor=geom.D12_ft))
    rows.append(dict(elemento="Lx_ORV,end [ft]", valor=geom.Lx_orv_end_ft))
    df_cent = pd.DataFrame(rows)
    st.dataframe(df_cent, use_container_width=True)
    st.download_button("Descargar tabla (CSV)", df_cent.to_csv(index=False).encode("utf-8"),
                       file_name="tabla_geometria_centros.csv", mime="text/csv")

# ===================== 7) GLOSARIO + SIGLAS =====================
with st.tabs(["Glosario + Siglas"])[0]:
    st.markdown("**Cómo usar:** esta es tu referencia rápida de símbolos y abreviaturas.")
    st.markdown("#### Símbolos (con unidades)")
    for k,v in SYMS.items():
        st.markdown(f"- ${v.latex}$ · **[{v.unit}]** — {v.desc}")
    st.markdown("#### Siglas y abreviaturas")
    for k,v in ABBR.items():
        st.markdown(f"- **{k}** — {v}")
