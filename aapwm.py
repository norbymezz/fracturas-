# spe215031_parallel_allin_v2.py
# SPE-215031-PA — Pozos paralelos: geometría por números + PDE→Laplace
# Presets (6 ejemplos) + verificaciones + foco y-ésimo + estilo configurable
# Ejecutá:  streamlit run spe215031_parallel_allin_v2.py

import math
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ================= UI base =================
st.set_page_config(page_title="SPE-215031-PA — Pozos paralelos (All-in v2)", layout="wide")
st.title("SPE-215031-PA — Interferencia de pozos paralelos (All-in v2)")

# ---------------- Colores sincronizados (chips ↔ cotas) ----------------
COLORS = {
    "spacing": "#0ea5e9",  # cyan-500
    "LxI":     "#22c55e",  # green-500
    "two_xO":  "#a855f7",  # violet-500 (2·x_O,i)
    "LxOint":  "#06b6d4",  # teal-500 (derivado)
    "total":   "#f59e0b",  # amber-500
    "dark":    "#111111",
}

def chip(label, color):
    return f"<span style='background:{color};color:white;padding:2px 6px;border-radius:6px;font-size:12px'>{label}</span>"

def colored_label(html: str):
    st.markdown(html, unsafe_allow_html=True)

def num_input_with_chip(label_html: str, key: str, **kwargs):
    colored_label(label_html)
    return st.number_input(" ", key=key, label_visibility="collapsed", **kwargs)

def checkbox_with_chip(label_html: str, key: str, value: bool = False):
    cols = st.columns([1, 6])
    with cols[1]:
        colored_label(label_html)
        return st.checkbox(" ", value=value, key=key, label_visibility="collapsed")

# --- estado de geometría aplicada: SIEMPRE leemos de acá para dibujar y calcular
if "geom_applied" not in st.session_state:
    st.session_state.geom_applied = dict(
        N=2,
        Lx_I_i_ft=[100.0, 100.0],      # [ft] por pozo
        two_xO_i_ft=[10.0, 10.0],      # [ft] 2·x_O,i por pozo
        spacing_g_ft=[400.0],          # [ft] spacing entre i–i+1
    )

# ================= Símbolos & Siglas =================
@dataclass
class Sym:
    latex: str; unit: str; desc: str
def _h(k): s=SYMS[k]; return f"${s.latex}$ · [{s.unit}] — {s.desc}"
SYMS: Dict[str, Sym] = {
    "mu":Sym(r"\mu","cP","Viscosidad del fluido"),
    "ct":Sym(r"c_t","1/psi","Compresibilidad total"),
    "h": Sym(r"h","ft","Espesor de la capa"),
    "p0":Sym(r"p_{\mathrm{res}}","psi","Presión inicial"),
    "kI":Sym(r"k_I","nD","Permeabilidad dominio I (SRV)"),
    "kO":Sym(r"k_O","nD","Permeabilidad dominio O (ORV)"),
    "LxI":Sym(r"L_{x,I}","ft","Semi-espesor SRV por pozo (transversal x)"),
    "two_xO":Sym(r"2x_{O,i}","ft","Ancho ORV extremo de la tabla (por pozo)"),
    "LxOend":Sym(r"L_{x,O,\mathrm{end}}","ft","Extensión ORV a extremos (x_{O,i}=½·2x_{O,i})"),
    "spacing":Sym(r"\mathrm{spacing}_g","ft","Separación entre ejes (hueco g)"),
    "LxOint":Sym(r"L_{x,O,\mathrm{int}}","ft","ORV entre pozos (derivado del spacing)"),
    "nf":Sym(r"n_f","–","# fracturas por pozo (en y)"),
    "xf":Sym(r"x_f","ft","Semi-espaciamiento de fracturas (en y)"),
}
ABBR = {"I":"Inner/SRV","O":"Outer/ORV","Pwf":"Well flowing pressure","DP":"Dual Porosity"}

# ================= Conversión Field↔SI =================
PSI_TO_PA=6894.757293168; FT_TO_M=0.3048; CP_TO_PAS=1e-3; ND_TO_M2=9.869233e-22
DAY_TO_S=86400.0; STB_TO_M3=0.158987294928
def si_mu(mu_cp): return mu_cp*CP_TO_PAS
def si_k(k_nD): return k_nD*ND_TO_M2
def si_ct(ct_invpsi): return ct_invpsi/PSI_TO_PA
def si_h(h_ft): return h_ft*FT_TO_M
def si_L(L_ft): return L_ft*FT_TO_M
def field_p(p_pa): return p_pa/PSI_TO_PA

# ================= Estabilidad =================
def tanh_stable(x):
    x=np.asarray(x,float); out=np.empty_like(x)
    big=x>20.0; sml=x<-20.0; mid=~(big|sml)
    out[big]=1.0; out[sml]=-1.0; out[mid]=np.tanh(x[mid]); return out
def coth_stable(x):
    x=np.asarray(x,float); ax=np.abs(x); out=np.empty_like(x)
    tiny=ax<1e-8; out[tiny]=1.0/x[tiny]+x[tiny]/3.0; out[~tiny]=1.0/tanh_stable(x[~tiny]); return out
def exp_clamped(z, lim=700.0): return np.exp(np.clip(z, -lim, lim))

# ================= Laplace: Stehfest & Gaver-Euler =================
if "fs_calls_stehfest" not in st.session_state: st.session_state.fs_calls_stehfest=0
if "fs_calls_gaver" not in st.session_state: st.session_state.fs_calls_gaver=0
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
    ln2=math.log(2.0); S=[]
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

def stehfest_nodes_weights(N:int, t:float):
    assert N%2==0 and N>0 and t>0
    V = stehfest_weights(N)
    s_nodes = (np.arange(1,N+1)*math.log(2.0))/t
    return s_nodes, V

# ================= Núcleo físico (Laplace) =================
def _lambda(mu, ct, k, s):  return math.sqrt(max(s,1e-40)*max(mu*ct,1e-40)/max(k,1e-40))

# DP (Eq. 30)
def f_dualpor(s: float, omega: float, Lambda: float) -> float:
    s=max(s,1e-40); om=max(min(omega,0.999999),1e-12); Lam=max(Lambda,1e-12)
    a = math.sqrt(Lam*(1.0-om)/(3.0*s)); b = math.sqrt(3.0*(1.0-om)*s/Lam)
    return om + a*math.tanh(b)

def R_slab_no_flow(mu, ct, k, h, L, s, use_dp=False, omega=0.5, Lambda=1.0):
    lam = _lambda(mu, ct, k, s); x = lam*max(L,1e-12)
    base = (mu/(k*max(h,1e-12))) * (coth_stable(x)/max(lam,1e-30))
    return f_dualpor(s,omega,Lambda)*base if use_dp else base

def R_semi_inf(mu, ct, k, h, s, use_dp=False, omega=0.5, Lambda=1.0):
    lam = _lambda(mu, ct, k, s); base = (mu/(k*max(h,1e-12))) * (1.0/max(lam,1e-30))
    return f_dualpor(s,omega,Lambda)*base if use_dp else base

def R_self(mu, ct, k_I, k_O, h, Lx_I, Lx_O_end, s,
           use_dp_I=False, omega_I=0.5, Lambda_I=1.0,
           use_dp_O=False, omega_O=0.5, Lambda_O=1.0):
    R_I = R_slab_no_flow(mu, ct, k_I, h, Lx_I, s, use_dp_I, omega_I, Lambda_I)
    R_O = R_slab_no_flow(mu, ct, k_O, h, Lx_O_end, s, use_dp_O, omega_O, Lambda_O) if Lx_O_end>0 \
          else R_semi_inf(mu, ct, k_O, h, s, use_dp_O, omega_O, Lambda_O)
    return R_I + R_O

def R_cross(mu, ct, k_O, h, Dij, s):
    lam_O = _lambda(mu, ct, k_O, s)
    return (mu/(k_O*max(h,1e-12))) * float(exp_clamped(-lam_O*max(Dij,0.0))) / max(lam_O,1e-30)

# ================= Schedules =================
# ================= Schedules =================
@dataclass
class WellSched: t_days: List[float]; q_stbd: List[float]
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
        tk = tk_day*DAY_TO_S; acc += (qk-last_q)*math.exp(-s*tk); last_q=qk
    return acc/max(s,1e-30)

# ================= Geometría por números =================
def build_geometry_numbers(spacing_g_ft: List[float], Lx_I_i_ft: List[float], two_xO_i_ft: List[float]):
    N = len(Lx_I_i_ft); assert len(two_xO_i_ft)==N and len(spacing_g_ft)==max(0,N-1)
    Lx_O_end_i_ft = [0.5*max(0.0, float(v)) for v in two_xO_i_ft]  # 2·x_O,i → x_O,i
    Lx_O_int_g_ft=[]; warnings=[]
    for g in range(max(0,N-1)):
        val = float(spacing_g_ft[g]) - (float(Lx_I_i_ft[g]) + float(Lx_I_i_ft[g+1]))
        if val < 0:
            warnings.append(f"Lx_O,int[{g+1}]<0; truncado a 0 (spacing={spacing_g_ft[g]}, Lx_I={Lx_I_i_ft[g]}+{Lx_I_i_ft[g+1]}).")
            val = 0.0
        Lx_O_int_g_ft.append(val)
    total_ft = Lx_O_end_i_ft[0] + sum(2.0*float(Lx_I_i_ft[i]) for i in range(N)) + sum(Lx_O_int_g_ft) + Lx_O_end_i_ft[-1]
    # D_ij por acumulación de spacings
    Dij_ft = np.zeros((N,N), float)
    cum = np.cumsum([0.0]+[float(s) for s in spacing_g_ft])
    for i in range(N):
        for j in range(N):
            Dij_ft[i,j] = abs(cum[j]-cum[i])
    return dict(Lx_O_end_i_ft=Lx_O_end_i_ft, Lx_O_int_g_ft=Lx_O_int_g_ft, total_ft=total_ft, Dij_ft=Dij_ft, warnings=warnings)

def schematic_numbers(
    Lx_I_i_ft,            # list[float]  -> por pozo
    two_xO_i_ft,          # list[float]  -> 2·x_O,i por pozo
    spacing_g_ft,         # list[float]  -> gaps entre pozos consecutivos
    Lx_O_int_g_ft,        # list[float]  -> Lx_O,int_g (derivado)
    total_ft: float,      # float        -> total entre bordes externos
    max_items: Optional[int] = None,
):
    """
    Esquema NO a escala:
      • Por pozo i: dibuja el POZO (barra) y muestra 2·Lx_I y 2·x_O,i con flechas.
      • Por gap g:  muestra spacing_g y Lx_O,int_g.
      • 'max_items' permite recortar (muestra primeras/últimas y '…' en el medio).
    Renderiza con plotly y lo envía a Streamlit.
    """

    # colores (si no hay un dict global, usamos estos por defecto)
    _C = {
        "well":    "#cc6f00",
        "LxI":     "#2ecc71",
        "two_xO":  "#9b59b6",
        "spacing": "#36a2eb",
        "LxOint":  "#ffd166",
        "total":   "#ff7f50",
        "text":    "#cfe8ff",
        "dots":    "#999999",
        "bg":      "#0b1220",
    }

    def sliced(n, k: Optional[int]):
        """Devuelve índices con posible recorte y flag si hubo '…'."""
        if (k is None) or (k <= 0) or (n <= k) or (k < 3):
            return list(range(n)), False
        head = max(1, k // 2)
        tail = k - head
        return list(range(head)) + ["…"] + list(range(n - tail, n)), True

    N = len(Lx_I_i_ft)
    idx_w, has_gap_w = sliced(N, max_items)
    idx_g, has_gap_g = sliced(max(0, N - 1), None if max_items is None else max(2, max_items - 1))

    fig = go.Figure()
    # ocultamos ejes y usamos coordenadas normalizadas en X (0..1)
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False)

    y0, dy = 1.0, 0.85

    # ---------- POZOS ----------
    row = 0
    for idx in idx_w:
        y = y0 + row * dy
        if idx == "…":
            fig.add_annotation(x=0.5, y=y, text="⋯", showarrow=False,
                               font=dict(size=16, color=_C["dots"]))
        else:
            i = idx
            # Pozo (barra vertical centrada)
            fig.add_shape(
                type="rect", x0=0.48, x1=0.52, y0=y - 0.18, y1=y + 0.18,
                line=dict(color=_C["well"], width=1),
                fillcolor=_C["well"], opacity=0.95,
            )
            # Etiqueta i
            fig.add_annotation(x=0.02, y=y, text=f"<b>i={i+1}</b>", showarrow=False,
                               xanchor="left", font=dict(size=11, color=_C["text"]))

            # Flecha 2·Lx_I (izquierda)
            fig.add_annotation(
                x=0.35, y=y, ax=0.05, ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, arrowhead=2, arrowwidth=1.2,
                arrowcolor=_C["LxI"]
            )
            fig.add_annotation(
                x=0.20, y=y-0.05,
                text=f"<span style='color:{_C['LxI']}'>2·Lx_I</span> = {2*Lx_I_i_ft[i]:.0f} ft",
                showarrow=False, xanchor="center", font=dict(size=11)
            )

            # Flecha 2·x_O,i (derecha)
            fig.add_annotation(
                x=0.65, y=y, ax=0.95, ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, arrowhead=2, arrowwidth=1.2,
                arrowcolor=_C["two_xO"]
            )
            fig.add_annotation(
                x=0.80, y=y-0.05,
                text=f"<span style='color:{_C['two_xO']}'>2·x_O,i</span> = {two_xO_i_ft[i]:.0f} ft",
                showarrow=False, xanchor="center", font=dict(size=11)
            )
        row += 1

    # ---------- GAPS ----------
    base = y0 + row * dy + 0.35
    r2 = 0
    for idx in idx_g:
        y = base + r2 * 0.7
        if idx == "…":
            fig.add_annotation(x=0.5, y=y, text="⋯", showarrow=False,
                               font=dict(size=16, color=_C["dots"]))
        else:
            g = idx
            fig.add_annotation(
                x=0.5, y=y,
                text=(f"<span style='color:{_C['spacing']}'>spacing_{g+1}</span> = {spacing_g_ft[g]:.0f} ft"
                      f"   →   <span style='color:{_C['LxOint']}'>Lx_O,int_{g+1}</span> = {Lx_O_int_g_ft[g]:.0f} ft"),
                showarrow=False, xanchor="center", font=dict(size=11)
            )
        r2 += 1

    # ---------- TOTAL ----------
    yT = base + r2 * 0.7 + 0.6
    fig.add_annotation(
        x=0.5, y=yT,
        text=f"<b style='color:{_C['total']}'>TOTAL</b> entre bordes externos = {total_ft:.0f} ft",
        showarrow=False, xanchor="center", font=dict(size=12)
    )

    # Layout
    extra = 36 if (has_gap_w or has_gap_g) else 0
    H = 110 + 26 * (row + r2) + extra
    fig.update_layout(
        height=H,
        margin=dict(l=10, r=10, t=6, b=6),
        paper_bgcolor=_C["bg"],
        plot_bgcolor=_C["bg"],
    )

    st.plotly_chart(fig, use_container_width=True)
# ================= Presets (6 ejemplos) + loader =================
EXAMPLES = {
    # Fig. tipo 4 pozos (valores de demo; ajustá si querés)
    "E1 · 4 pozos (460|500|440), 2·x_O=(10,60,100,40), Lx_I=100": {
        "props": dict(mu_cp=1.0, k_I_nD=800.0, k_O_nD=150.0, ct_invpsi=1.0e-6, h_ft=65.6, p0_psi=4350.0),
        "geom": dict(N=4, Lx_I_i_ft=[100,100,100,100], two_xO_i_ft=[10,60,100,40], spacing_g_ft=[460,500,440],
                     expected_total_ft=1820.0),
        "wells": [
            dict(nf=25, xf=200.0, t_days=[0,10,30], q_stbd=[1000,800,0]),
            dict(nf=25, xf=200.0, t_days=[0,10,30], q_stbd=[0,500,0]),
            dict(nf=25, xf=200.0, t_days=[0],       q_stbd=[0]),
            dict(nf=25, xf=200.0, t_days=[0],       q_stbd=[0]),
        ]
    },

    "E2 · 2 pozos simétricos (spacing=600), 2·x_O=(50,50), Lx_I=120": {
        "props": dict(mu_cp=1.2, k_I_nD=500.0, k_O_nD=200.0, ct_invpsi=8e-7, h_ft=80.0, p0_psi=3500.0),
        "geom": dict(N=2, Lx_I_i_ft=[120,120], two_xO_i_ft=[50,50], spacing_g_ft=[600],
                     expected_total_ft=50/2 + 240 + (600-240) + 50/2),
        "wells": [
            dict(nf=21, xf=180.0, t_days=[0,5,20], q_stbd=[800,600,0]),
            dict(nf=21, xf=180.0, t_days=[0],      q_stbd=[0]),
        ]
    },

    "E3 · 3 pozos (Lx_I=[80,100,120]), spacing=[500,450], 2·x_O=(20,60,40)": {
        "props": dict(mu_cp=0.8, k_I_nD=1000.0, k_O_nD=120.0, ct_invpsi=1.2e-6, h_ft=70.0, p0_psi=4200.0),
        "geom": dict(N=3, Lx_I_i_ft=[80,100,120], two_xO_i_ft=[20,60,40], spacing_g_ft=[500,450],
                     expected_total_ft=None),
        "wells": [
            dict(nf=27, xf=160.0, t_days=[0,2,15,40], q_stbd=[500,900,600,0]),
            dict(nf=25, xf=200.0, t_days=[0,10,30],   q_stbd=[0,400,0]),
            dict(nf=25, xf=220.0, t_days=[0],         q_stbd=[0]),
        ]
    },

    "E4 · 4 pozos (spacing=[350,350,350]), 2·x_O=(30,30,30,30), Lx_I=140": {
        "props": dict(mu_cp=1.0, k_I_nD=700.0, k_O_nD=100.0, ct_invpsi=9e-7, h_ft=60.0, p0_psi=3500.0),
        "geom": dict(N=4, Lx_I_i_ft=[140]*4, two_xO_i_ft=[30,30,30,30], spacing_g_ft=[350,350,350],
                     expected_total_ft=None),
        "wells": [
            dict(nf=24, xf=190.0, t_days=[0,10,25], q_stbd=[600,600,0]),
            dict(nf=24, xf=190.0, t_days=[0,10,25], q_stbd=[0,300,0]),
            dict(nf=24, xf=190.0, t_days=[0],       q_stbd=[0]),
            dict(nf=24, xf=190.0, t_days=[0],       q_stbd=[0]),
        ]
    },

    "E5 · 5 pozos (child tardío), spacing=[400,420,440,460], Lx_I=100": {
        "props": dict(mu_cp=1.1, k_I_nD=900.0, k_O_nD=160.0, ct_invpsi=1e-6, h_ft=65.6, p0_psi=4300.0),
        "geom": dict(N=5, Lx_I_i_ft=[100]*5, two_xO_i_ft=[20,40,60,40,20], spacing_g_ft=[400,420,440,460],
                     expected_total_ft=None),
        "wells": [
            dict(nf=25, xf=200.0, t_days=[0,60],    q_stbd=[900,0]),
            dict(nf=25, xf=200.0, t_days=[0],       q_stbd=[0]),
            dict(nf=25, xf=200.0, t_days=[0],       q_stbd=[0]),
            dict(nf=25, xf=200.0, t_days=[0,30,90], q_stbd=[0,600,0]),
            dict(nf=25, xf=200.0, t_days=[0],       q_stbd=[0]),
        ]
    },

    "E6 · 3 pozos (extremos grandes), spacing=[450,450], 2·x_O=(120,40,120), Lx_I=110": {
        "props": dict(mu_cp=0.9, k_I_nD=600.0, k_O_nD=140.0, ct_invpsi=1.1e-6, h_ft=75.0, p0_psi=4100.0),
        "geom": dict(N=3, Lx_I_i_ft=[110,110,110], two_xO_i_ft=[120,40,120], spacing_g_ft=[450,450],
                     expected_total_ft=None),
        "wells": [
            dict(nf=26, xf=210.0, t_days=[0,5,50],  q_stbd=[700,700,0]),
            dict(nf=26, xf=210.0, t_days=[0],       q_stbd=[0]),
            dict(nf=26, xf=210.0, t_days=[0,20,60], q_stbd=[0,500,0]),
        ]
    },
}

def load_example(name: str):
    ex = EXAMPLES[name]
    # props
    for k,v in ex["props"].items(): st.session_state[k] = v
    # geom
    G = ex["geom"]; N = G["N"]; st.session_state["N_from_example"] = N
    for i,val in enumerate(G["Lx_I_i_ft"]): st.session_state[f"LxI_{i}"] = float(val)
    for i,val in enumerate(G["two_xO_i_ft"]): st.session_state[f"2xOi_{i}"] = float(val)
    for i,val in enumerate(G["spacing_g_ft"]): st.session_state[f"sp_{i}"] = float(val)
    st.session_state["expected_total_ft"] = G.get("expected_total_ft", None)
    # wells
    if "wells" not in st.session_state: st.session_state.wells=[]
    while len(st.session_state.wells) < N:
        st.session_state.wells.append(dict(nf=25, xf=200.0, sched=WellSched([0.0],[0.0])))
    while len(st.session_state.wells) > N: st.session_state.wells.pop()
    for i,wd in enumerate(ex["wells"]):
        st.session_state.wells[i]["nf"] = int(wd["nf"])
        st.session_state.wells[i]["xf"] = float(wd["xf"])
        st.session_state.wells[i]["sched"] = ensure_schedule(WellSched(wd["t_days"], wd["q_stbd"]))


# ================= Verificaciones =================
def verify_geometry_totals(Lx_I_i_ft, two_xO_i_ft, spacing_g_ft, expected_total_ft=None):
    G = build_geometry_numbers(spacing_g_ft, Lx_I_i_ft, two_xO_i_ft)
    ok = (expected_total_ft is None) or (abs(G["total_ft"] - expected_total_ft) < 1e-6)
    return ok, G["total_ft"], G["warnings"]

def verify_schedules_monotonic(wells) -> list:
    msgs=[]
    for i,w in enumerate(wells):
        t=w["sched"].t_days
        if any(t[j+1] < t[j] for j in range(len(t)-1)):
            msgs.append(f"Schedule i={i+1}: tiempos no monótonos.")
    return msgs

# ================= Estado inicial (wells vacíos) =================
if "wells" not in st.session_state:
    st.session_state.wells = [dict(nf=25, xf=200.0, sched=WellSched([0.0],[0.0])) for _ in range(4)]

# ================= Tabs =================
tabs = st.tabs([
    "1) Propiedades (I/O)", "2) Geometría (números)", "3) Pozos & Schedules",
    "4) PDE → Laplace → Álgebra", "Elemento en foco", "5) Resultados", "6) Tablas", "Glosario"
])
tab_props, tab_geom, tab_sched, tab_pde, tab_focus, tab_res, tab_tbl, tab_gloss = tabs

# -------- 1) Propiedades --------
with tab_props:
    st.markdown("Cargá propiedades de fluido/roca y permeabilidades por dominio **K∈{I,O}**.")
    # ================= NUEVO LAYOUT =================
    # Propiedades (izquierda) + Doble porosidad (derecha)

    colL, colR = st.columns([1,1])

    # ---------------------- IZQUIERDA ----------------------
    with colL:
        st.markdown("### Propiedades fluido/roca")
        st.markdown("<div style='font-size:12px'>", unsafe_allow_html=True)

        mu_cp   = st.number_input(_h("mu"), 0.01, 50.0,
                                  float(st.session_state.get("mu_cp", 1.0)), 0.01, key="mu_cp")
        k_I_nD  = st.number_input(_h("kI"), 0.1, 1e6,
                                  float(st.session_state.get("k_I_nD", 800.0)), 0.1, key="k_I_nD")
        k_O_nD  = st.number_input(_h("kO"), 0.1, 1e6,
                                  float(st.session_state.get("k_O_nD", 150.0)), 0.1, key="k_O_nD")
        ct_invpsi = st.number_input(_h("ct"), 1e-7, 1e-2,
                                    float(st.session_state.get("ct_invpsi", 1.0e-6)), format="%.2e", key="ct_invpsi")
        h_ft     = st.number_input(_h("h"), 1.0, 300.0,
                                   float(st.session_state.get("h_ft", 65.6)), 0.1, key="h_ft")
        p0_psi   = st.number_input(_h("p0"), 100.0, 10000.0,
                                   float(st.session_state.get("p0_psi", 4350.0)), 1.0, key="p0_psi")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- DERECHA ----------------------
    with colR:
        st.markdown("### Doble porosidad (opcional)")
        st.markdown("<div style='font-size:12px'>", unsafe_allow_html=True)

        use_dp_I = st.checkbox("DP en I (SRV)", value=False, key="use_dp_I")
        omega_I  = st.slider("ω_I (fracción conectada)", 0.0, 1.0, 0.3, 0.01,
                             key="omega_I", disabled=not use_dp_I)
        Lambda_I = st.slider("Λ_I (relación de movilidades)", 1e-4, 1e2, 1.0,
                             0.01, format="%.3g", key="Lambda_I", disabled=not use_dp_I)

        use_dp_O = st.checkbox("DP en O (ORV)", value=False, key="use_dp_O")
        omega_O  = st.slider("ω_O (fracción conectada)", 0.0, 1.0, 0.3, 0.01,
                             key="omega_O", disabled=not use_dp_O)
        Lambda_O = st.slider("Λ_O (relación de movilidades)", 1e-4, 1e2, 1.0,
                             0.01, format="%.3g", key="Lambda_O", disabled=not use_dp_O)

        st.markdown("""
        **Interpretación rápida:**  
        - **ω**: fracción de la porosidad conectada al flujo rápido. Si ω→1, el medio tiende a simple porosidad.  
        - **Λ**: velocidad de intercambio matriz–fractura.  
          Λ grande ⇒ acoplamiento rápido; Λ pequeño ⇒ retardo (almacenamiento).
        """)

        st.markdown("</div>", unsafe_allow_html=True)

# -------- 2) Geometría (números) --------
with tab_geom:
    st.markdown("Cargá spacing, 2·x_O,i y Lx_I. Usá **Aplicar geometría** para actualizar.")
    # ===== Geometría (FORM) =====
    with st.form("geom_form"):
        G = st.session_state.geom_applied  # alias
        cA, cB = st.columns(2)

        # --- Columna A: N y Lx_I ---
        with cA:
            ui_N = st.number_input(
                "N pozos paralelos", 2, 12, int(G["N"]), 1,
                help="Cantidad de pozos paralelos (arreglo en planta).",
                key="ui_N",
            )
            ui_uniform = st.checkbox(
                "Lx_I uniforme", True,
                help="Usar el mismo Lx_I para todos los pozos.",
                key="ui_LxI_uniform",
            )
            if ui_uniform:
                ui_LxI_all = st.number_input(
                    "Lx_I [ft] (todos los pozos)", 1.0, 5e4, float(G["Lx_I_i_ft"][0]), 1.0,
                    help="Semi-longitud característica SRV por pozo (perpendicular al pozo).",
                    key="ui_LxI_all",
                )
                ui_LxI_i = [ui_LxI_all] * ui_N
            else:
                ui_LxI_i = [
                    st.number_input(
                        f"Lx_I [ft] (i={i+1})", 1.0, 5e4,
                        float(G["Lx_I_i_ft"][i] if i < len(G["Lx_I_i_ft"]) else 100.0),
                        1.0,
                        help="Semi-longitud SRV del pozo i.",
                        key=f"ui_LxI_{i}",
                    )
                    for i in range(ui_N)
                ]

        # --- Columna B: 2·x_O,i y spacing ---
        with cB:
            ui_two_xO_i = [
                st.number_input(
                    f"2·x_O,i [ft] (i={i+1})", 0.0, 2e5,
                    float(G["two_xO_i_ft"][i] if i < len(G["two_xO_i_ft"]) else 10.0),
                    1.0,
                    help="Ancho total del ORV en extremos del pozo i (dos laterales).",
                    key=f"ui_2xOi_{i}",
                )
                for i in range(ui_N)
            ]
            ui_spacing = [
                st.number_input(
                    f"spacing_{g+1} [ft] (entre i={g+1}–i={g+2})", 1.0, 2e5,
                    float(G["spacing_g_ft"][g] if g < len(G["spacing_g_ft"]) else 400.0),
                    1.0,
                    help="Distancia entre ejes de pozos consecutivos (gaps).",
                    key=f"ui_sp_{g}",
                )
                for g in range(max(0, ui_N - 1))
            ]

        apply_geom = st.form_submit_button("Aplicar geometría", use_container_width=True)

    # >>> Al hacer click en Aplicar, persistimos SOLO en geom_applied (no tocamos keys de widgets)
    if apply_geom:
        st.session_state.geom_applied = dict(
            N=int(ui_N),
            Lx_I_i_ft=[float(v) for v in ui_LxI_i],
            two_xO_i_ft=[float(v) for v in ui_two_xO_i],
            spacing_g_ft=[float(v) for v in ui_spacing],
        )
        st.success("Geometría aplicada.")

   # ===== Valores vigentes (SIEMPRE desde geom_applied) =====
G = st.session_state.geom_applied
N = int(G["N"])
Lx_I_i_ft   = list(G["Lx_I_i_ft"])
two_xO_i_ft = list(G["two_xO_i_ft"])
spacing_g_ft= list(G["spacing_g_ft"])

Gnums = build_geometry_numbers(spacing_g_ft, Lx_I_i_ft, two_xO_i_ft)  # tu helper

with st.expander("Cotas / esquema (opcional)", expanded=False):
    max_items = st.slider("Recorte de filas (vista compacta)", 0, 12, 0, help="0 = sin recorte")
    show_cotas = st.checkbox("Mostrar esquema con cotas (no a escala)", True)
    if show_cotas:
        schematic_numbers(
            Lx_I_i_ft, two_xO_i_ft, spacing_g_ft,
            Gnums["Lx_O_int_g_ft"], Gnums["total_ft"],
            max_items if max_items > 0 else None,
        )

# (opcional) verificación de total si cargaste expected_total_ft en algún ejemplo
meta_total = st.session_state.get("expected_total_ft", None)
if meta_total is not None:
    ok, total_now, warnG = verify_geometry_totals(Lx_I_i_ft, two_xO_i_ft, spacing_g_ft, meta_total)
    for m in warnG: st.warning(m)
    st.success(f"TOTAL = {total_now:.0f} ft") if ok else st.error(f"TOTAL = {total_now:.0f} ft (esperado {meta_total:.0f})")

# -------- 3) Pozos & Schedules --------
with tab_sched:
    st.markdown("Editá **nf, x_f** y el schedule por pozo (columnas `t_days` y `q_stbd` separadas por coma).")
    # armamos DataFrame compacto
    N = len(st.session_state.wells)
    data = []
    for i,w in enumerate(st.session_state.wells):
        data.append(dict(
            pozo=i+1,
            nf=int(w.get("nf",20)),
            x_f=float(w.get("xf",200.0)),
            t_days=", ".join(str(x) for x in w["sched"].t_days),
            q_stbd=", ".join(str(x) for x in w["sched"].q_stbd),
        ))
    df0 = pd.DataFrame(data)

    with st.form("sched_form"):
        df_edit = st.data_editor(
            df0, use_container_width=True,
            column_config={
                "pozo": st.column_config.NumberColumn("pozo", disabled=True),
                "nf":   st.column_config.NumberColumn("n_f", min_value=1, max_value=300, step=1),
                "x_f":  st.column_config.NumberColumn("x_f [ft]", min_value=1.0, max_value=3000.0, step=1.0),
                "t_days": st.column_config.TextColumn("t_ℓ [day] (coma)"),
                "q_stbd": st.column_config.TextColumn("q_ℓ [STB/D] (coma)"),
            },
            hide_index=True
        )
        ok_apply = st.form_submit_button("Aplicar schedules/pozos", use_container_width=True)

    if ok_apply:
        try:
            for r in df_edit.itertuples(index=False):
                i = int(r.pozo)-1
                st.session_state.wells[i]["nf"] = int(r.nf)
                st.session_state.wells[i]["xf"] = float(r.x_f)
                t_vals = [float(x.strip()) for x in str(r.t_days).split(",") if str(x).strip()!=""]
                q_vals = [float(x.strip()) for x in str(r.q_stbd).split(",") if str(x).strip()!=""]
                st.session_state.wells[i]["sched"] = ensure_schedule(WellSched(t_vals, q_vals))
            st.success("Pozos y schedules actualizados.")
        except Exception as e:
            st.error(f"Error al parsear: {e}")

    bad_sched = verify_schedules_monotonic(st.session_state.wells)
    if bad_sched: [st.error(m) for m in bad_sched]

# -------- 4) PDE → Laplace → Álgebra --------
with tab_pde:
    st.markdown("### Del PDE 1D (transversal x) al sistema algebraico en Laplace")
    st.markdown("""
Consideramos, por bloque **K∈{I,O}**, un *slab* 1D en **x** con no-flux en los bordes (o semi-infinito en O si no hay borde).  
Ecuación de difusión con fuente lineal en el eje del pozo (presión promedio de línea):
""")
    st.latex(r"""
\frac{\partial p_K}{\partial t}
= \frac{k_K}{\mu\,c_t}\,\frac{\partial^2 p_K}{\partial x^2}
\;+\; \frac{q_i(t)}{A_K}\,\delta(x)
\qquad \text{con}\quad \left.\frac{\partial p_K}{\partial x}\right|_{\pm L_{x,K}}=0
""")
    st.markdown("Aplicando **Laplace** en \(t\) (con \(p_K(x,0)=p_{\rm res}\)):")
    st.latex(r"""
s\,\hat p_K(x,s) - p_{\rm res}
= \frac{k_K}{\mu\,c_t}\,\frac{\mathrm d^2\hat p_K}{\mathrm d x^2}
\;+\; \frac{\hat q_i(s)}{A_K}\,\delta(x)
""")
    st.markdown("La solución con condiciones Neumann da una relación lineal **presión-promedio en el eje** ↔ **caudal**:")
    st.latex(r"""
\hat p_{i,K}(s) - p_{\rm res}/s
= \left[\frac{\mu}{k_K\,h}\,\frac{\coth(\lambda_K L_{x,K})}{\lambda_K}\right]\hat q_i(s)
\quad,\quad
\lambda_K=\sqrt{\frac{\mu c_t}{k_K}\,s}
""")
    st.markdown("Sumando contribuciones de **I** y **O** (propio) y añadiendo la influencia **cruzada** a través de **O** a distancia \(D_{ij}\):")
    st.latex(r"""
\hat p_i(s)=\sum_{j} R_{ij}(s)\,\hat q_j(s),
\quad
R_{ii}=\frac{\mu}{h}\Big[\frac{1}{k_I}\frac{\coth(\lambda_I L_{x,I})}{\lambda_I}+\frac{1}{k_O}\frac{\coth(\lambda_O L_{x,O,end})}{\lambda_O}\Big],
\quad
R_{ij}=\frac{\mu}{k_O h}\frac{e^{-\lambda_O D_{ij}}}{\lambda_O}.
""")
    st.markdown("**DP opcional (Eq. 30)**: multiplicamos el término de *slab* por \(f_K(s)\).")
    st.latex(r"""f_K(s)=\omega_K+\sqrt{\frac{\Lambda_K(1-\omega_K)}{3s}}\tanh\!\left(\sqrt{\frac{3(1-\omega_K)s}{\Lambda_K}}\right)""")
    st.markdown("La inversión de Laplace la hacemos con **Stehfest** o **Gaver–Euler** para obtener \(Pwf_i(t)=p_{\rm res}-\mathcal{L}^{-1}\{\hat p_i\}(t)\).")

# -------- Elemento en foco (y-ésimo) --------
def R_matrix_at_s(mu,ct,kI,kO,h, LxI_list_SI, LxOend_list_SI, Dij_SI, s,
                  use_dp_I,omega_I,Lambda_I, use_dp_O,omega_O,Lambda_O):
    n=len(LxI_list_SI); R=np.zeros((n,n),float)
    for i in range(n):
        R[i,i]=R_self(mu,ct,kI,kO,h, LxI_list_SI[i], LxOend_list_SI[i], s,
                      use_dp_I,omega_I,Lambda_I, use_dp_O,omega_O,Lambda_O)
    for i in range(n):
        for j in range(n):
            if i!=j: R[i,j]=R_cross(mu,ct,kO,h, Dij_SI[i,j], s)
    conv = (1/PSI_TO_PA) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)
    return R*conv

def qhat_vector_at_s(s, schedules):
    return np.array([q_hat_piecewise_s(ws, s) for ws in schedules], float)

with tab_focus:
    st.markdown("### Elemento en foco — superposición en Laplace e inversión Stehfest")
    st.caption("Elegí i (destino), j (fuente), un tiempo t* y el nodo k. Se muestran R_ij(s*), q̂_j(s*), el aporte y la descomposición Stehfest.")

    Nw = len(st.session_state.wells)
    c1,c2,c3,c4 = st.columns(4)
    with c1: i_sel = st.number_input("i (destino)", 1, Nw, 1, 1) - 1
    with c2: j_sel = st.number_input("j (fuente)", 1, Nw, min(2,Nw), 1) - 1
    with c3: t_star_day = st.number_input("t* [day]", 1e-6, 1e6, 10.0, format="%.6f")
    with c4: Nst_focus = st.slider("N Stehfest", 8, 20, 12, 2, key="Nst_focus")
    t_star = t_star_day*DAY_TO_S
    s_nodes, V = stehfest_nodes_weights(Nst_focus, t_star)
    k_idx = st.slider("k (nodo)", 1, Nst_focus, min(6,Nst_focus), 1)
    s_star = s_nodes[k_idx-1]

    # Geometría/props actuales → listas SI
    Lx_I_i_ft = [st.session_state.get(f"LxI_{i}", 100.0) for i in range(Nw)]
    two_xO_i_ft = [st.session_state.get(f"2xOi_{i}", 10.0) for i in range(Nw)]
    spacing_g_ft = [st.session_state.get(f"sp_{i}", 400.0) for i in range(max(0,Nw-1))]
    Gcur = build_geometry_numbers(spacing_g_ft, Lx_I_i_ft, two_xO_i_ft)

    LI_SI = [si_L(x) for x in Lx_I_i_ft]
    LOe_SI = [si_L(x) for x in Gcur["Lx_O_end_i_ft"]]
    Dij_SI = si_L(Gcur["Dij_ft"])
    mu,ct,h = si_mu(st.session_state.get("mu_cp",1.0)), si_ct(st.session_state.get("ct_invpsi",1e-6)), si_h(st.session_state.get("h_ft",65.6))
    kI,kO = si_k(st.session_state.get("k_I_nD",800.0)), si_k(st.session_state.get("k_O_nD",150.0))
    use_dp_I = st.session_state.get("use_dp_I", False); omega_I=st.session_state.get("omega_I",0.3); Lambda_I=st.session_state.get("Lambda_I",1.0)
    use_dp_O = st.session_state.get("use_dp_O", False); omega_O=st.session_state.get("omega_O",0.3); Lambda_O=st.session_state.get("Lambda_O",1.0)
    Scheds = [ensure_schedule(w["sched"]) for w in st.session_state.wells]

    R_at_s = R_matrix_at_s(mu,ct,kI,kO,h, LI_SI,LOe_SI, Dij_SI, s_star,
                           use_dp_I,omega_I,Lambda_I, use_dp_O,omega_O,Lambda_O)  # [psi·day/STB]
    qhat_s = qhat_vector_at_s(s_star, Scheds)                                     # [STB]
    pihat_s = R_at_s.dot(qhat_s)                                                  # [psi·day]
    Rij = R_at_s[i_sel, j_sel]; qj = qhat_s[j_sel]; contrib = Rij*qj

    st.markdown(
        f"{chip('s*', COLORS['dark'])} = {s_star:.3e} 1/s &nbsp;&nbsp; "
        f"{chip('R_ij(s*)', COLORS['dark'])} = {Rij:.3e} [psi·day/STB] &nbsp;&nbsp; "
        f"{chip('q̂_j(s*)', COLORS['dark'])} = {qj:.3e} [STB] &nbsp;&nbsp; "
        f"{chip('aporte = R_ij·q̂_j', COLORS['dark'])} = {contrib:.3e} [psi·day] &nbsp;&nbsp; "
        f"{chip('p̂_i(s*) = Σ_j', COLORS['dark'])} = {pihat_s[i_sel]:.3e} [psi·day]",
        unsafe_allow_html=True
    )

    # Barras: contribuciones Stehfest
# Barras: contribuciones Stehfest
    pihats = []
    for s_k in s_nodes:
        Rk = R_matrix_at_s(mu,ct,kI,kO,h, LI_SI,LOe_SI, Dij_SI, s_k,
                           use_dp_I,omega_I,Lambda_I, use_dp_O,omega_O,Lambda_O)
        qhk = qhat_vector_at_s(s_k, Scheds)
        pihats.append( (Rk.dot(qhk))[i_sel] )
    pihats = np.array(pihats)
    terms = (math.log(2.0)/t_star) * V * pihats  # contribuciones por k
    figB = go.Figure(data=[go.Bar(x=[f"k={k}" for k in range(1,Nst_focus+1)], y=terms)])
    figB.update_layout(height=260, title="Descomposición Stehfest de p_i(t*) (términos por nodo)",
                       yaxis_title="[psi]")
    st.plotly_chart(figB, use_container_width=True)

    # Kernel cruzado vs distancia (para s*)
    Ds_ft = np.linspace(0.0, max(1.0, max(Gcur["Dij_ft"].flatten())*1.05), 80)
    Ds_SI = si_L(Ds_ft)
    lamO = _lambda(mu,ct,kO,s_star)
    Rcross_vsD = (mu/(kO*max(h,1e-12))) * np.exp(-lamO*Ds_SI) / max(lamO,1e-30) * (1/PSI_TO_PA) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)
    figK = go.Figure()
    figK.add_scatter(x=Ds_ft, y=Rcross_vsD, mode="lines", name="R_cross(s*, D)")
    figK.add_vline(x=Gcur["Dij_ft"][i_sel,j_sel], line=dict(dash="dash"))
    figK.update_layout(height=260, title="Kernel cruzado R_ij(s*, D) vs D  (línea punteada: D_ij actual)",
                       xaxis_title="D [ft]", yaxis_title="R [psi·day/STB]")
    st.plotly_chart(figK, use_container_width=True)

# -------- 5) Resultados --------
# -------- 5) Resultados (TabRez) --------
with tab_res:
    with st.form("results_form"):
        st.markdown("Configurá estilo, método de inversión y rango temporal. Presioná **Calcular y graficar** para actualizar.")
        # --- Estilo ---
        with st.expander("Estilo de gráficos", expanded=False):
            log_x = st.checkbox("X log", True)
            log_y_P = st.checkbox("Y log (P, ΔP)", False)
            log_y_q = st.checkbox("Y log (q)", False)
            line_dash = st.selectbox("Trazo", ["solid","dash","dot","dashdot"], 0)
            show_markers = st.checkbox("Marcadores", True)
            marker_symbol = st.selectbox("Símbolo", ["circle","square","triangle-up","diamond","cross","x","pentagon","star"], 0)
            marker_size = st.slider("Tamaño marcador", 4, 16, 7, 1)
            line_width = st.slider("Espesor línea", 1, 6, 2, 1)

        # --- Inversión de Laplace ---
        c1,c2,c3 = st.columns(3)
        with c1: inv_m = st.radio("Inversión", ["Stehfest","Gaver–Euler"], 0)
        with c2: Nst   = st.slider("N (Stehfest)", 8, 20, 12, 2)
        with c3:
            M_gv = st.slider("M (Gaver)", 12, 40, 18, 2)
            P_gv = st.slider("P (Euler)", 6, 14, 8, 1)

        # --- Tiempo ---
        c4,c5,c6 = st.columns(3)
        with c4: tmin_d = st.number_input("t_min [day]", 1e-6, 1e2, 1e-3, format="%.6f")
        with c5: tmax_d = st.number_input("t_max [day]", 1e-4, 1e6, 1e3, format="%.4f")
        with c6: npts   = st.number_input("n_pts", 16, 2000, 220, 1)

        do_run = st.form_submit_button("Calcular y graficar", use_container_width=True)

    if not do_run:
        st.info("Ajustá parámetros y presioná **Calcular y graficar**.")
    else:
        # ---------- Preparación ----------
        times_s = np.logspace(np.log10(tmin_d), np.log10(tmax_d), int(npts))*DAY_TO_S

        # props SI
        mu  = si_mu(st.session_state.get('mu_cp', 1.0))
        kI  = si_k(st.session_state.get('k_I_nD', 800.0))
        kO  = si_k(st.session_state.get('k_O_nD', 150.0))
        ct  = si_ct(st.session_state.get('ct_invpsi', 1.0e-6))
        h   = si_h(st.session_state.get('h_ft', 65.6))
        p0_psi = float(st.session_state.get('p0_psi', 4350.0))

        # DP (opcional)
        use_dp_I = st.session_state.get("use_dp_I", False); omega_I=st.session_state.get("omega_I",0.3); Lambda_I=st.session_state.get("Lambda_I",1.0)
        use_dp_O = st.session_state.get("use_dp_O", False); omega_O=st.session_state.get("omega_O",0.3); Lambda_O=st.session_state.get("Lambda_O",1.0)

        # geometría actual
        Nw = len(st.session_state.wells)
        Lx_I_i_ft  = [st.session_state.get(f"LxI_{i}", 100.0) for i in range(Nw)]
        two_xO_i_ft= [st.session_state.get(f"2xOi_{i}", 10.0) for i in range(Nw)]
        spacing_g_ft = [st.session_state.get(f"sp_{i}", 400.0) for i in range(max(0,Nw-1))]
        G = build_geometry_numbers(spacing_g_ft, Lx_I_i_ft, two_xO_i_ft)

        # conversiones
        LI  = [si_L(x) for x in Lx_I_i_ft]
        LOe = [si_L(x) for x in G["Lx_O_end_i_ft"]]
        Dij = si_L(G["Dij_ft"])

        # schedules asegurados
        WS = [ensure_schedule(w["sched"]) for w in st.session_state.wells]

        # ---------- F(s): devuelve \hat p(s) en [psi·day] ----------
        def Fvec(s: float)->np.ndarray:
            R_SI = np.zeros((Nw,Nw), float)
            # propio (I + O)
            for i in range(Nw):
                R_SI[i,i] = R_self(mu, ct, kI, kO, h, LI[i], LOe[i], s,
                                   use_dp_I=use_dp_I, omega_I=omega_I, Lambda_I=Lambda_I,
                                   use_dp_O=use_dp_O, omega_O=omega_O, Lambda_O=Lambda_O)
            # cruzado (O)
            for i in range(Nw):
                for j in range(Nw):
                    if i!=j: R_SI[i,j] = R_cross(mu, ct, kO, h, Dij[i,j], s)
            # q̂(s) por pozo
            qhat = np.array([q_hat_piecewise_s(ws, s) for ws in WS], float)  # [STB]
            # convertir: [Pa·s]→[psi·day]
            conv = (1/PSI_TO_PA) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)
            return (R_SI*conv).dot(qhat)

        # ---------- Inversión ----------
        st.session_state.fs_calls_stehfest=0; st.session_state.fs_calls_gaver=0
        if inv_m=="Stehfest":
            Pwf = np.vstack([ p0_psi - invert_stehfest_vec(Fvec, ti, Nst) for ti in times_s ])
        else:
            Pwf = np.vstack([ p0_psi - invert_gaver_euler_vec(Fvec, ti, M=M_gv, P=P_gv) for ti in times_s ])

        # ---------- Helpers de estilo ----------
        def line_kwargs():
            return dict(mode="lines+markers" if show_markers else "lines",
                        line=dict(width=line_width, dash=line_dash),
                        marker=dict(symbol=marker_symbol, size=marker_size))

        # ---------- Gráficos ----------
        # Pwf
        with st.expander("Pwf_i(t)", expanded=True):
            fig=go.Figure()
            for i in range(Nw):
                fig.add_scatter(x=times_s/DAY_TO_S, y=Pwf[:,i], name=f"i={i+1}: Pwf [psi]", **line_kwargs())
            fig.update_xaxes(title="t [day]", type="log" if log_x else "linear")
            fig.update_yaxes(title="Pwf [psi]", type="log" if log_y_P else "linear")
            fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        # ΔP_res = p_res − Pwf
        with st.expander("ΔP_res,i(t) = p_res − Pwf_i(t)"):
            fig=go.Figure()
            for i in range(Nw):
                fig.add_scatter(x=times_s/DAY_TO_S, y=p0_psi-Pwf[:,i], name=f"i={i+1}: ΔP_res [psi]", **line_kwargs())
            fig.update_xaxes(title="t [day]", type="log" if log_x else "linear")
            fig.update_yaxes(title="ΔP_res [psi]", type="log" if log_y_P else "linear")
            fig.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        # ΔP_ij entre vecinos (j=i+1)
        if Nw>=2:
            with st.expander("ΔP_{ij}(t) = Pwf_j − Pwf_i (vecinos)"):
                fig=go.Figure()
                for i in range(Nw-1):
                    j=i+1
                    fig.add_scatter(x=times_s/DAY_TO_S, y=Pwf[:,j]-Pwf[:,i], name=f"ΔP_{i+1}{j+1} [psi]", **line_kwargs())
                fig.update_xaxes(title="t [day]", type="log" if log_x else "linear")
                fig.update_yaxes(title="ΔP_{ij} [psi]", type="log" if log_y_P else "linear")
                fig.update_layout(height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

        # q_i(t) (del schedule)
        with st.expander("q_i(t) (desde schedules)"):
            tmax_all = max([w["sched"].t_days[-1] if w["sched"].t_days else 1.0 for w in st.session_state.wells] + [1.0])
            ts=np.linspace(0,tmax_all,400)
            fig=go.Figure()
            for i,w in enumerate(st.session_state.wells):
                # escalonado por tramos
                tarr=np.array(w["sched"].t_days); qarr=np.array(w["sched"].q_stbd)
                qs=np.zeros_like(ts)
                for m,ti in enumerate(ts):
                    k=np.searchsorted(tarr, ti, side="right")-1; k=max(k,0)
                    qs[m]=qarr[k]
                fig.add_scatter(x=ts, y=qs, name=f"i={i+1}: q [STB/D]", **line_kwargs())
            fig.update_xaxes(title="t [day]", type="log" if log_x else "linear")
            fig.update_yaxes(title="q [STB/D]", type="log" if log_y_q else "linear")
            fig.update_layout(height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Llamadas a F(s) — Stehfest: {st.session_state.fs_calls_stehfest} · Gaver(Euler): {st.session_state.fs_calls_gaver}")
# -------- 6) Tablas --------
with tab_tbl:
    rows=[]
    rows.append(dict(tipo="geom", name="N", value=len(st.session_state.wells)))
    for i in range(len(st.session_state.wells)):
        rows.append(dict(tipo="pozo", name=f"2·Lx_I (i={i+1}) [ft]", value=2*st.session_state.get(f"LxI_{i}", 100.0)))
        rows.append(dict(tipo="pozo", name=f"2·x_O,i (i={i+1}) [ft]", value=st.session_state.get(f"2xOi_{i}", 10.0)))
    for g in range(max(0,len(st.session_state.wells)-1)):
        rows.append(dict(tipo="hueco", name=f"spacing_{g+1} [ft]", value=st.session_state.get(f"sp_{g}", 400.0)))
    Gtab = build_geometry_numbers(
        [st.session_state.get(f"sp_{i}", 400.0) for i in range(max(0,len(st.session_state.wells)-1))],
        [st.session_state.get(f"LxI_{i}", 100.0) for i in range(len(st.session_state.wells))],
        [st.session_state.get(f"2xOi_{i}", 10.0) for i in range(len(st.session_state.wells))]
    )
    for g,val in enumerate(Gtab["Lx_O_int_g_ft"], start=1):
        rows.append(dict(tipo="hueco", name=f"Lx_O,int_{g} [ft] (derivado)", value=val))
    rows.append(dict(tipo="geom", name="TOTAL [ft]", value=Gtab["total_ft"]))
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

# -------- Glosario --------
with tab_gloss:
    st.markdown("**Símbolos**")
    for k,v in SYMS.items(): st.markdown(f"- ${v.latex}$ · **[{v.unit}]** — {v.desc}")
    st.markdown("---")
    st.markdown("**Siglas** — " + " · ".join([f"**{k}**: {v}" for k,v in ABBR.items()]))