
# spe_unified_app.py
# Ejecutar:  streamlit run spe_unified_app.py
# Requisitos: streamlit, plotly, numpy
# Integra y generaliza ideas de aaapp.py, aapwm.py y speFiguresAndPlots.html,
# y agrega un flujo unificado para resolver 4 "application examples" de:
#  - SPE 102834-PA (Medeiros-Ozkan-Kazemi, bloques heterogeneos)
#  - SPE 125043-PA (Trilinear SRV/ORV, Colorado)

import math, json, io, csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import os, sys

# Integrate physics kernels from frac-modeling-main if available
_BASE_DIR = os.path.dirname(__file__)
_FRAC_MOD_DIR = os.path.join(_BASE_DIR, 'frac-modeling-main', 'frac-modeling-main')
if os.path.isdir(_FRAC_MOD_DIR) and _FRAC_MOD_DIR not in sys.path:
    sys.path.append(_FRAC_MOD_DIR)

try:
    from physics import (
        tanh_stable as _tanh_stable,
        coth_stable as _coth_stable,
        exp_clamped as _exp_clamped,
        invert_stehfest_vec as _invert_stehfest_vec,
        invert_gaver_euler_vec as _invert_gaver_euler_vec,
        R_slab_no_flow as _R_slab_no_flow,
        R_semi_inf as _R_semi_inf,
        R_cross as _R_cross,
        R_self as _R_self,
    )
    _HAS_FRAC_PHYSICS = True
except Exception:
    _HAS_FRAC_PHYSICS = False

# ===================== UI base =====================
st.set_page_config(page_title="SPE — Ejemplos Unificados (102834 & 125043)", layout="wide")
st.title("SPE — Ejemplos Unificados (102834-PA & 125043-PA)")

# ===================== Utilidades =====================
PSI_TO_PA=6894.757293168
FT_TO_M=0.3048
CP_TO_PAS=1e-3
MD_TO_M2=9.869233e-16
ND_TO_M2=9.869233e-22
DAY_TO_S=86400.0
STB_TO_M3=0.158987294928

def si_mu(mu_cp): return mu_cp*CP_TO_PAS
def si_k_md(k_md): return k_md*MD_TO_M2
def si_k_nd(k_nd): return k_nd*ND_TO_M2
def si_ct(ct_invpsi): return ct_invpsi/PSI_TO_PA
def si_h(h_ft): return h_ft*FT_TO_M
def si_L(L_ft): return L_ft*FT_TO_M
def field_p(p_pa): return p_pa/PSI_TO_PA

def tanh_stable(x):
    if _HAS_FRAC_PHYSICS:
        return _tanh_stable(x)
    x=np.asarray(x,float); out=np.empty_like(x)
    big=x>20.0; sml=x<-20.0; mid=~(big|sml)
    out[big]=1.0; out[sml]=-1.0; out[mid]=np.tanh(x[mid]); return out
def coth_stable(x):
    if _HAS_FRAC_PHYSICS:
        return _coth_stable(x)
    x=np.asarray(x,float); ax=np.abs(x); out=np.empty_like(x)
    tiny=ax<1e-8; out[tiny]=1.0/x[tiny]+x[tiny]/3.0; out[~tiny]=1.0/tanh_stable(x[~tiny]); return out
def exp_clamped(z, lim=700.0):
    if _HAS_FRAC_PHYSICS:
        return _exp_clamped(z, lim)
    return np.exp(np.clip(z, -lim, lim))

# ===================== Inversion de Laplace =====================
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

def invert_stehfest_vec(Fvec: Callable[[float], np.ndarray], t: float, N:int=12)->np.ndarray:
    if _HAS_FRAC_PHYSICS:
        return _invert_stehfest_vec(Fvec, t, N)
    if t<=0: return np.nan
    V=stehfest_weights(N)
    s_nodes = (np.arange(1,N+1)*math.log(2.0))/max(t,1e-30)
    vals=[np.asarray(Fvec(s), float) for s in s_nodes]
    vals=np.stack(vals, axis=0)
    return (math.log(2.0)/t) * (V[:,None]*vals).sum(axis=0)

# ===================== Nucleo fisico (resistencias en Laplace) =====================
def _lambda(mu, ct, k, s):  return math.sqrt(max(s,1e-40)*max(mu*ct,1e-40)/max(k,1e-40))

def R_slab_no_flow(mu, ct, k, h, L, s):
    if _HAS_FRAC_PHYSICS:
        return _R_slab_no_flow(mu, ct, k, h, L, s)
    lam = _lambda(mu, ct, k, s); x = lam*max(L,1e-12)
    return (mu/(k*max(h,1e-12))) * (coth_stable(x)/max(lam,1e-30))

def R_semi_inf(mu, ct, k, h, s):
    if _HAS_FRAC_PHYSICS:
        return _R_semi_inf(mu, ct, k, h, s)
    lam = _lambda(mu, ct, k, s); return (mu/(k*max(h,1e-12))) * (1.0/max(lam,1e-30))

def f_dualpor(s: float, omega: float, Lambda: float) -> float:
    s=max(s,1e-40); om=max(min(omega,0.999999),1e-12); Lam=max(Lambda,1e-12)
    a = math.sqrt(Lam*(1.0-om)/(3.0*s)); b = math.sqrt(3.0*(1.0-om)*s/Lam)
    return om + a*math.tanh(b)

def R_slab_no_flow_DP(mu, ct, k, h, L, s, omega=0.5, Lambda=1.0):
    if _HAS_FRAC_PHYSICS:
        return _R_slab_no_flow(mu, ct, k, h, L, s, True, omega, Lambda)
    return f_dualpor(s,omega,Lambda)*R_slab_no_flow(mu,ct,k,h,L,s)

def R_cross_exp(mu, ct, k, h, Dij, s):
    if _HAS_FRAC_PHYSICS:
        return _R_cross(mu, ct, k, h, Dij, s)
    lam = _lambda(mu, ct, k, s)
    return (mu/(k*max(h,1e-12))) * float(exp_clamped(-lam*max(Dij,0.0))) / max(lam,1e-30)

# ===================== Schedules =====================
@dataclass
class WellSched:
    t_days: List[float]
    q_stbd: List[float]
def ensure_schedule(ws: 'WellSched')->'WellSched':
    if not ws.t_days: ws.t_days=[0.0]
    if not ws.q_stbd: ws.q_stbd=[0.0]
    pairs=sorted(zip(ws.t_days, ws.q_stbd), key=lambda z:z[0])
    t=[pairs[0][0]]; q=[pairs[0][1]]
    for ti,qi in pairs[1:]:
        if ti!=t[-1]: t.append(ti); q.append(qi)
        else: q[-1]=qi
    return WellSched(t_days=t, q_stbd=q)
def q_hat_piecewise_s(ws: 'WellSched', s: float) -> float:
    acc=0.0; last_q=0.0
    for tk_day, qk in zip(ws.t_days, ws.q_stbd):
        tk = tk_day*DAY_TO_S; acc += (qk-last_q)*math.exp(-s*tk); last_q=qk
    return acc/max(s,1e-30)

# ===================== Modelos =====================
def Fvec_trilinear_SRV_ORV(nW:int, mu_cp: float, ct_invpsi: float, h_ft: float,
                           k_SRV_md: float, Lx_SRV_ft: float,
                           k_ORV_md: float, Lx_ORV_end_ft: float,
                           spacing_between_wells_ft: float,
                           omega_I: float=1.0, Lambda_I: float=1.0,
                           omega_O: float=1.0, Lambda_O: float=1.0,
                           schedules: Optional[List[WellSched]]=None):
    # Basado en 125043-PA: SRV + ORV y acoplo lateral por ORV
    mu=si_mu(mu_cp); ct=si_ct(ct_invpsi); h=si_h(h_ft)
    kI=si_k_md(k_SRV_md); kO=si_k_md(k_ORV_md)
    Ls=si_L(Lx_SRV_ft); Lo=si_L(Lx_ORV_end_ft)
    Dij = np.zeros((nW,nW))
    for i in range(nW):
        for j in range(nW):
            Dij[i,j]=0.0 if i==j else abs(i-j)*spacing_between_wells_ft
    Dij = si_L(Dij)

    schedules = schedules or [WellSched([0.0],[200.0]) for _ in range(nW)]
    schedules = [ensure_schedule(ws) for ws in schedules]

    def Fvec(s: float)->np.ndarray:
        R = np.zeros((nW,nW), float)
        for i in range(nW):
            Ri = R_slab_no_flow(mu,ct,kI,h,Ls,s) if omega_I>=0.999 else R_slab_no_flow_DP(mu,ct,kI,h,Ls,s,omega_I,Lambda_I)
            Ro = R_slab_no_flow(mu,ct,kO,h,Lo,s) if Lx_ORV_end_ft>0 else R_semi_inf(mu,ct,kO,h,s)
            if omega_O<0.999: Ro = R_slab_no_flow_DP(mu,ct,kO,h,Lo,s,omega_O,Lambda_O)
            R[i,i]=Ri+Ro
        for i in range(nW):
            for j in range(nW):
                if i!=j:
                    R[i,j]=R_cross_exp(mu,ct,kO,h,Dij[i,j],s)
        qhat_STB = np.array([q_hat_piecewise_s(ws, s) for ws in schedules], float)
        p_hat_SI = R.dot(qhat_STB * STB_TO_M3 / DAY_TO_S)     # [Pa*s]
        return field_p(p_hat_SI) * DAY_TO_S                    # [psi*day]
    return Fvec

@dataclass
class Block:
    Lx_ft: float
    k_md: float
    dp_omega: float = 1.0
    dp_Lambda: float = 1.0

def Fvec_blocked_horizontal(nW:int, mu_cp: float, ct_invpsi: float, h_ft: float,
                            blocks: List[Block],
                            spacing_between_wells_ft: float,
                            schedules: Optional[List[WellSched]]=None):
    # Basado en 102834-PA: suma de resistencias 1D por bloques + acoplo lateral
    mu=si_mu(mu_cp); ct=si_ct(ct_invpsi); h=si_h(h_ft)
    Dij = np.zeros((nW,nW))
    for i in range(nW):
        for j in range(nW):
            Dij[i,j]=0.0 if i==j else abs(i-j)*spacing_between_wells_ft
    Dij = si_L(Dij)
    schedules = schedules or [WellSched([0.0],[200.0]) for _ in range(nW)]
    schedules = [ensure_schedule(ws) for ws in schedules]

    def Fvec(s: float)->np.ndarray:
        R = np.zeros((nW,nW), float)
        Rii = 0.0
        for b in blocks:
            k = si_k_md(b.k_md); L = si_L(b.Lx_ft)
            term = R_slab_no_flow(mu,ct,k,h,L,s) if b.dp_omega>=0.999 else R_slab_no_flow_DP(mu,ct,k,h,L,s,b.dp_omega,b.dp_Lambda)
            Rii += term
        for i in range(nW): R[i,i]=Rii
        k_ext = si_k_md(blocks[-1].k_md if len(blocks)>0 else 10.0)
        for i in range(nW):
            for j in range(nW):
                if i!=j:
                    R[i,j]=R_cross_exp(mu,ct,k_ext,h,Dij[i,j],s)
        qhat_STB = np.array([q_hat_piecewise_s(ws, s) for ws in schedules], float)
        p_hat_SI = R.dot(qhat_STB * STB_TO_M3 / DAY_TO_S)
        return field_p(p_hat_SI) * DAY_TO_S
    return Fvec

# ===================== Panel de control =====================
with st.sidebar:
    st.header("Configuracion")
    model_kind = st.selectbox("Modelo", ["125043-PA · Trilinear (SRV/ORV)",
                                         "102834-PA · Bloques (heterogeneo)"])
    N_steh = st.select_slider("N (Stehfest, par)", options=[8,10,12,14,16], value=12)
    t_min = st.number_input("t_min [day]", value=1e-3, step=1e-3, format="%.3g")
    t_max = st.number_input("t_max [day]", value=1000.0, step=1.0)
    n_pts = st.number_input("n_pts", value=220, step=1)
    st.caption("Usa los presets para arrancar rapido.")
    table_mode = st.selectbox("Tabla de resultados", ["Streamlit", "Plotly Table"], index=0)

# ===================== Presets =====================
st.subheader("Presets rapidos (4 Application Examples)")
colA, colB, colC, colD = st.columns(4)
preset_name = None
if colA.button("102834-PA · App.Ex. 2"):
    preset_name = "102834_EX2"
if colB.button("102834-PA · App.Ex. 3"):
    preset_name = "102834_EX3"
if colC.button("125043-PA · Ex. A"):
    preset_name = "125043_EXA"
if colD.button("125043-PA · Ex. B"):
    preset_name = "125043_EXB"

def load_presets():
    try:
        with open("presets_examples.json","r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
preset_db = load_presets()
if preset_name and preset_name in preset_db:
    st.success(f"Preset cargado: {preset_name}")
else:
    if preset_name:
        st.warning("Preset no encontrado; usa la configuracion actual.")

# ===================== Inputs comunes =====================
st.markdown("### Propiedades de fluido y formacion (Field)")
c1,c2,c3,c4 = st.columns(4)
mu_cp = c1.number_input("mu [cP]", value=0.6, step=0.01)
ct_invpsi = c2.number_input("c_t [1/psi]", value=8e-6, step=1e-6, format="%.1e")
h_ft = c3.number_input("h [ft]", value=40.0, step=1.0)
p_res = c4.number_input("p_res [psi]", value=4000.0, step=1.0)

st.markdown("### Pozos & schedule")
c1,c2 = st.columns(2)
nW = int(c1.number_input("N pozos", value=1, min_value=1, step=1))
spacing = c2.number_input("D_ij entre pozos contiguos [ft]", value=450.0, step=1.0)

schedules = []
for i in range(nW):
    with st.expander(f"Pozo {i+1} — Schedule (STB/d)", expanded=(i==0)):
        t_list = st.text_input(f"t_dias (coma - Pozo {i+1})", value="0, 1, 10, 100")
        q_list = st.text_input(f"q_STB/d (coma - Pozo {i+1})", value="200, 200, 100, 0")
        try:
            t_vals = [float(x.strip()) for x in t_list.split(",") if x.strip()!=""]
            q_vals = [float(x.strip()) for x in q_list.split(",") if x.strip()!=""]
        except Exception:
            t_vals, q_vals = [0.0],[200.0]
        schedules.append(WellSched(t_vals, q_vals))

# === Parametros especificos de cada modelo ===
if model_kind.startswith("125043"):
    st.markdown("### Parametros SRV/ORV (125043-PA)")
    c1,c2,c3 = st.columns(3)
    k_SRV_md = c1.number_input("k_SRV [md]", value=500.0, step=10.0)
    Lx_SRV_ft = c2.number_input("Lx_SRV [ft] (semi-largo)", value=100.0, step=1.0)
    Lx_ORV_end_ft = c3.number_input("Lx_ORV,end [ft]", value=200.0, step=1.0)
    c1,c2,c3,c4 = st.columns(4)
    k_ORV_md = c1.number_input("k_ORV [md]", value=10.0, step=1.0)
    omega_I = c2.number_input("omega_I (DP SRV)", value=1.0, step=0.05)
    Lambda_I = c3.number_input("Lambda_I (DP SRV)", value=1000.0, step=10.0)
    omega_O = c4.number_input("omega_O (DP ORV)", value=1.0, step=0.05)
    Lambda_O = st.number_input("Lambda_O (DP ORV)", value=1000.0, step=10.0)
else:
    st.markdown("### Bloques en x (102834-PA)")
    nB = st.number_input("# de bloques", value=3, min_value=1, step=1)
    blocks: List[Block] = []
    for b in range(nB):
        c1,c2,c3,c4 = st.columns(4)
        Lx_ft_b = c1.number_input(f"Bloque {b+1} — Lx [ft]", value=100.0 if b==0 else 300.0, step=0.1)
        k_md_b = c2.number_input(f"Bloque {b+1} — k [md]", value=10.0, step=0.1)
        omg_b = c3.number_input(f"Bloque {b+1} — omega (DP)", value=1.0, step=0.05)
        Lam_b = c4.number_input(f"Bloque {b+1} — Lambda (DP)", value=1000.0, step=10.0)
        blocks.append(Block(Lx_ft_b, k_md_b, omg_b, Lam_b))

# ===================== Ejecutar =====================
run = st.button("Recalcular curvas Pwf(t)")
if run:
    if model_kind.startswith("125043"):
        Fvec = Fvec_trilinear_SRV_ORV(nW, mu_cp, ct_invpsi, h_ft,
                                      k_SRV_md, Lx_SRV_ft, k_ORV_md, Lx_ORV_end_ft,
                                      spacing, omega_I, Lambda_I, omega_O, Lambda_O, schedules)
    else:
        Fvec = Fvec_blocked_horizontal(nW, mu_cp, ct_invpsi, h_ft, blocks, spacing, schedules)

    ts = np.geomspace(max(t_min,1e-6), max(t_max,1.0), int(n_pts))
    pwf_all = [invert_stehfest_vec(Fvec, t, N_steh) for t in ts]
    P = np.vstack(pwf_all)  # [n_times, nW]  psi

    fig = go.Figure()
    for i in range(nW):
        fig.add_trace(go.Scatter(x=ts, y=P[:,i], mode="lines", name=f"Pwf pozo {i+1}"))
    fig.update_layout(xaxis_type="log", xaxis_title="t [day]", yaxis_title="Pwf [psi]",
                      template="plotly_dark", height=520)
    st.plotly_chart(fig, use_container_width=True)

    data_dict = {"t_day": ts}
    for i in range(nW):
        data_dict[f"Pwf_{i+1}_psi"] = P[:, i]

    if table_mode == "Plotly Table":
        headers = list(data_dict.keys())
        cells = [data_dict[k] for k in headers]
        tbl = go.Figure(data=[go.Table(header=dict(values=headers), cells=dict(values=cells))])
        tbl.update_layout(height=400)
        st.plotly_chart(tbl, use_container_width=True)
    else:
        st.dataframe(data_dict)

    # CSV download
    output = io.StringIO()
    writer = csv.writer(output)
    headers = list(data_dict.keys())
    writer.writerow(headers)
    for r in range(len(ts)):
        writer.writerow([data_dict[k][r] for k in headers])
    st.download_button(
        label="Descargar CSV",
        data=output.getvalue().encode("utf-8"),
        file_name="pwf_curvas.csv",
        mime="text/csv"
    )

# ===================== Rubrica =====================
st.markdown("""
### Criterio de evaluacion (pedagogico y funcional)
Se calcula una meta heuristica A–E con cinco dimensiones (0–5 c/u):
1. Trazabilidad teorica: parametros visibles y unidades consistentes.
2. Estabilidad numerica: inversion de Laplace configurable; tiempos log-espaciados.
3. Flexibilidad: cantidad de pozos, bloques y dual porosity (omega, Lambda).
4. Claridad de interfaz: secciones y nombres estandar.
5. Reproducibilidad: presets documentados y tablas exportables.

Una calificacion "Excelente" requiere >= 22/25.
""")
