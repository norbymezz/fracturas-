# app_spe215031_allinone_v5.py
# Ejecutar: streamlit run app_spe215031_allinone_v5.py
# SOLO SPE-215031-PA. Sin motores externos.
# v5: diferencias Apilado vs Lateral (texto+fórmulas+SVG dinámico),
#     editar/eliminar pozos, álgebra ligada a las fórmulas del paper.

import os, math, base64
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ====== UI base ======
st.set_page_config(page_title="SPE-215031-PA — Interferencia (desde cero)", layout="wide")
st.title("SPE-215031-PA — Interferencia de pozos horizontales (desde cero)")

# ====== Glosario ======
@dataclass
class Sym:
    latex: str; unit: str; desc: str
def _h(k): s=SYMS[k]; return f"${s.latex}$ · [{s.unit}] — {s.desc}"

SYMS: Dict[str, Sym] = {
    "mu":   Sym(r"\mu", "cP", "Viscosidad del fluido"),
    "k":    Sym(r"k", "nD", "Permeabilidad efectiva"),
    "ct":   Sym(r"c_t", "1/psi", "Compresibilidad total"),
    "h":    Sym(r"h", "ft", "Espesor de la capa"),
    "xF":   Sym(r"x_F", "ft", "Semi-espaciamiento entre fracturas"),
    "nF":   Sym(r"n_F", "–",  "Número de fracturas por pozo"),
    "LxSRV":    Sym(r"L_{x,\mathrm{SRV}}", "ft", "Semi-longitud característica SRV (flujo 1D)"),
    "LxORVin":  Sym(r"L_{x,\mathrm{ORV,int}}", "ft", "Separación patrón entre pozos (apilado)"),
    "LxORVend": Sym(r"L_{x,\mathrm{ORV,end}}", "ft", "Extensión ORV a extremos"),
    "p0":       Sym(r"p_{\mathrm{res}}", "psi", "Presión inicial"),
}

# ====== Conversión Field↔SI ======
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

# ====== Estabilidad ======
def tanh_stable(x):
    x=np.asarray(x,float); out=np.empty_like(x)
    big=x>20.0; sml=x<-20.0; mid=~(big|sml)
    out[big]=1.0; out[sml]=-1.0; out[mid]=np.tanh(x[mid]); return out
def coth_stable(x):
    x=np.asarray(x,float); ax=np.abs(x); out=np.empty_like(x)
    tiny=ax<1e-8; out[tiny]=1.0/x[tiny]+x[tiny]/3.0; out[~tiny]=1.0/tanh_stable(x[~tiny]); return out
def exp_clamped(z, lim=700.0): return np.exp(np.clip(z, -lim, lim))

# ====== Inversión de Laplace ======
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

# ====== Núcleo físico (Laplace) ======
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
def R_cross_lateral(mu, ct, k_orv, h, Dij, s):
    lam = _lambda(mu, ct, k_orv, s)
    return (mu/(k_orv*max(h,1e-12))) * float(exp_clamped(-lam*max(Dij,0.0))) / max(lam,1e-30)

# ====== Pozo & schedule ======
@dataclass
class Well:
    x: float; y: float; nF: int; xF: float
    t_days: List[float]; q_stbd: List[float]

def ensure_schedule(w: Well):
    if not w.t_days: w.t_days=[0.0]
    if not w.q_stbd: w.q_stbd=[0.0]
    pairs=sorted(zip(w.t_days, w.q_stbd), key=lambda z:z[0])
    t=[pairs[0][0]]; q=[pairs[0][1]]
    for ti,qi in pairs[1:]:
        if ti!=t[-1]: t.append(ti); q.append(qi)
        else: q[-1]=qi
    w.t_days, w.q_stbd = t, q

def q_hat_piecewise_s(w: Well, s: float) -> float:
    acc=0.0; last_q=0.0
    for tk_day, qk in zip(w.t_days, w.q_stbd):
        tk = tk_day*DAY_TO_S
        acc += (qk-last_q)*math.exp(-s*tk)
        last_q=qk
    return acc/max(s,1e-30)  # [STB]

# ====== Distancias ORV ======
def distances_apilado(wells: List[Well], Lx_orv_int_ft: float)->np.ndarray:
    n=len(wells); D=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j]=0.0 if i==j else abs(i-j)*Lx_orv_int_ft
    return D
def distances_lateral(wells: List[Well])->np.ndarray:
    n=len(wells); D=np.zeros((n,n))
    xs=[w.x for w in wells]
    for i in range(n):
        for j in range(n):
            D[i,j]=abs(xs[j]-xs[i])
    return D

# ====== Fvec(s) ======
def make_Fvec(wells: List[Well],
              mu_cp: float, k_srv_nD: float, k_orv_nD: float, ct_invpsi: float, h_ft: float,
              Lx_srv_ft: float, Lx_orv_end_ft: float,
              mode_orv: str, Lx_orv_int_ft: float):
    mu  = si_mu(mu_cp); kSRV= si_k(k_srv_nD); kORV= si_k(k_orv_nD)
    ct  = si_ct(ct_invpsi); h=si_h(h_ft); Ls=si_L(Lx_srv_ft); Lo=si_L(Lx_orv_end_ft)
    D_ft = distances_apilado(wells, Lx_orv_int_ft) if mode_orv=="Apilado (vertical)" else distances_lateral(wells)
    D = si_L(D_ft)
    def Fvec(s: float)->np.ndarray:
        n=len(wells); R = np.zeros((n,n), float)
        for i in range(n): R[i,i] = R_self(mu, ct, kSRV, kORV, h, Ls, Lo, s)
        for i in range(n):
            for j in range(n):
                if i!=j: R[i,j] = R_cross_lateral(mu, ct, kORV, h, D[i,j], s)
        qhat_STB = np.array([q_hat_piecewise_s(w, s) for w in wells], float)
        p_hat_SI = R.dot(qhat_STB * STB_TO_M3 / DAY_TO_S)     # [Pa·s]
        return field_p(p_hat_SI) * DAY_TO_S                    # [psi·day]
    return Fvec

# ====== PDF viewer ======
def find_pdf_candidate(fname_hint="spe-215031", folder="."):
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    for f in pdfs:
        if fname_hint in f.lower(): return os.path.join(folder, f)
    return os.path.join(folder, pdfs[0]) if pdfs else ""
def embed_pdf_bytes(data: bytes, height=720):
    b64 = base64.b64encode(data).decode("utf-8")
    st.components.v1.html(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px"></iframe>', height=height)

# ====== SVG helpers (con etiquetas de segmentos) ======
# ==== Etiquetas "Eq. (xx)" ====
# === Formato y “chips” de valores mostrados junto a las fórmulas ===
def fmt(v, unit, sig=3):
    try:
        return f"{float(v):.{sig}g} {unit}"
    except Exception:
        return f"{v} {unit}"

def chip(label, value, unit, sig=3):
    val = fmt(value, unit, sig)
    return (f"<span style='background:#1f2a44;color:#cfe8ff;"
            f"padding:2px 6px;border-radius:6px;margin-right:6px;"
            f"font-size:12px;white-space:nowrap'>{label}={val}</span>")

def chips_row(items):  # items = [(label, value, unit, sig), ...]
    return " ".join(chip(*i) if len(i)==3 else chip(*i[:3], i[3]) for i in items)

def _eqnum(key: str) -> str:
    """Devuelve 'xx' si fue cargado en la UI, o '' si está vacío."""
    val = str(st.session_state.get(f"eq_{key}", "")).strip()
    return val

def show_eq(key: str, where: str = "SPE-215031-PA") -> None:
    """Muestra 'Eq. (xx) — SPE-215031-PA' debajo de la fórmula si hay número."""
    num = _eqnum(key)
    if num:
        st.caption(f"Eq. ({num}) — {where}")

def _svg_header(W,H): 
    return [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="#0b1220" rx="10"/>']

def svg_orv_apilado(N, Lx_srv_ft, Lx_orv_int_ft, Lx_orv_end_ft):
    W,H = 960, 280 + 40*max(1,N)
    col_outer="#89c2ff"; col_inner="#3fb5ff"; col_well="#cc6f00"; col_txt="#cfe8ff"; col_dim="#ffd166"
    svg=_svg_header(W,H)
    svg.append(f'<text x="20" y="24" fill="{col_txt}" font-size="14">Apilado (vertical): N pozos ⇒ N+1 outers</text>')
    block_h=40; y0=60
    for i in range(2*N+1):
        y=y0+i*block_h; is_inner=(i%2==1)
        fill = col_inner if is_inner else col_outer
        label = (f"I{(i//2)+1}" if is_inner else f"O{(i//2)+1}")
        svg.append(f'<rect x="70" y="{y}" width="{W-140}" height="{block_h-10}" fill="{fill}" rx="6"/>')
        svg.append(f'<text x="80" y="{y+block_h//2}" fill="#0b1220" font-size="12" font-weight="600">{label}</text>')
        if is_inner and (i//2)<N:
            cy=y+(block_h-10)//2
            svg.append(f'<rect x="90" y="{cy-2}" width="{W-180}" height="4" fill="{col_well}" rx="2"/>')
    y_c = H-60
    svg += [
        f'<line x1="140" y1="{y_c}" x2="300" y2="{y_c}" stroke="{col_dim}" stroke-width="2"/>',
        f'<line x1="140" y1="{y_c-5}" x2="140" y2="{y_c+5}" stroke="{col_dim}" stroke-width="2"/>',
        f'<line x1="300" y1="{y_c-5}" x2="300" y2="{y_c+5}" stroke="{col_dim}" stroke-width="2"/>',
        f'<text x="156" y="{y_c-10}" fill="{col_dim}" font-size="12">2·Lx_SRV = {2*Lx_srv_ft:.1f} ft</text>',
        f'<line x1="340" y1="{y_c}" x2="520" y2="{y_c}" stroke="{col_dim}" stroke-width="2"/>',
        f'<line x1="340" y1="{y_c-5}" x2="340" y2="{y_c+5}" stroke="{col_dim}" stroke-width="2"/>',
        f'<line x1="520" y1="{y_c-5}" x2="520" y2="{y_c+5}" stroke="{col_dim}" stroke-width="2"/>',
        f'<text x="356" y="{y_c-10}" fill="{col_dim}" font-size="12">Lx_ORV,int = {Lx_orv_int_ft:.1f} ft</text>',
        f'<line x1="560" y1="{y_c}" x2="740" y2="{y_c}" stroke="{col_dim}" stroke-width="2"/>',
        f'<line x1="560" y1="{y_c-5}" x2="560" y2="{y_c+5}" stroke="{col_dim}" stroke-width="2"/>',
        f'<line x1="740" y1="{y_c-5}" x2="740" y2="{y_c+5}" stroke="{col_dim}" stroke-width="2"/>',
        f'<text x="572" y="{y_c-10}" fill="{col_dim}" font-size="12">Lx_ORV,end = {Lx_orv_end_ft:.1f} ft</text>',
    ]
    svg.append('</svg>')
    return "\n".join(svg)

def svg_orv_lateral(wells, Lx_srv_ft, Lx_orv_end_ft, show_labels=True):
    N=len(wells)
    W,H = 960, 320
    col_orv="#89c2ff"; col_srv="#3fb5ff"; col_well="#cc6f00"; col_txt="#cfe8ff"; col_dim="#ffd166"; col_tick="#ffffff"
    svg=_svg_header(W,H)
    svg.append(f'<text x="20" y="24" fill="{col_txt}" font-size="14">Lateral explícito: N pozos ⇒ 2N−1 segmentos ORV</text>')
    y_mid=170
    xs=[w.x for w in wells] if N>0 else [0.0]
    x_min=min(xs); x_max=max(xs); xL=x_min - Lx_orv_end_ft; xR=x_max + Lx_orv_end_ft
    def X(u): return 100 if xR==xL else int(100 + (W-200)*(u-xL)/(xR-xL))
    svg.append(f'<rect x="{X(xL)}" y="{y_mid-12}" width="{X(xR)-X(xL)}" height="24" fill="{col_orv}" rx="8"/>')
    for i,w in enumerate(wells):
        cx=X(w.x); wpx=max(6, int(2*Lx_srv_ft*(W-200)/(xR-xL))); x0=cx - wpx//2
        svg.append(f'<rect x="{x0}" y="{y_mid-20}" width="{wpx}" height="40" fill="{col_srv}" rx="6"/>')
        svg.append(f'<circle cx="{cx}" cy="{y_mid}" r="4" fill="{col_well}"/>')
        svg.append(f'<text x="{cx}" y="{y_mid+36}" fill="{col_txt}" font-size="12" text-anchor="middle">w{i+1}</text>')
    xs_sorted = sorted(xs); boundaries = [xL] + xs_sorted + [xR]
    for i in range(len(boundaries)-1):
        a, b = boundaries[i], boundaries[i+1]; Ax, Bx = X(a), X(b); mid = (Ax+Bx)//2; length = b - a
        svg.append(f'<line x1="{Ax}" y1="{y_mid-26}" x2="{Ax}" y2="{y_mid+26}" stroke="{col_tick}" stroke-width="2" opacity="0.8"/>')
        if show_labels: svg.append(f'<text x="{mid}" y="{y_mid-28}" fill="{col_dim}" font-size="11" text-anchor="middle">seg{i+1}: {length:.1f} ft</text>')
    svg.append(f'<line x1="{X(xR)}" y1="{y_mid-26}" x2="{X(xR)}" y2="{y_mid+26}" stroke="{col_tick}" stroke-width="2" opacity="0.8"/>')
    svg.append('</svg>')
    return "\n".join(svg)

# ====== Sidebar ======
with st.sidebar:
    st.markdown("---")
    st.subheader("Números de ecuación (PDF)")
    st.caption("Cargá los 'Eq. (xx)' exactos del SPE-215031-PA para que aparezcan junto a las fórmulas.")
    st.text_input("Eq. q̂(s) (por tramos)",  key="eq_qhat")
    st.text_input("Eq. λ (lambda)",         key="eq_lambda")
    st.text_input("Eq. R_slab",             key="eq_Rslab")
    st.text_input("Eq. R_semiinf",          key="eq_Rsemi")
    st.text_input("Eq. R_ij (interferencia)", key="eq_Rij")
    st.text_input("Eq. sistema p̂ = R·q̂",  key="eq_system")

    st.subheader("SPE-215031-PA (PDF)")
    default_guess = find_pdf_candidate()
    pdf_path = st.text_input("Ruta PDF", default_guess, key="pdf_path")
    upl = st.file_uploader("…o subí el PDF", type=["pdf"], key="pdf_upl")
    show_pdf = st.checkbox("Mostrar PDF a la derecha", True, key="pdf_show")

    st.markdown("---")
    st.subheader("Inversión de Laplace")
    inv_method = st.radio("Método", ["Stehfest (Gaver–Stehfest)","Gaver (Euler)"], index=0, key="inv_m")
    Nst = st.slider("N (Stehfest)", 8, 20, 12, 2, key="inv_N")
    M_gv = st.slider("M (Gaver)", 12, 40, 18, 2, key="inv_M")
    P_gv = st.slider("P (Euler)", 6, 14, 8, 1, key="inv_P")

# ====== Tabs ======
tabs = st.tabs([
    "1) Introduction",
    "2) Pozos & schedules (editar/eliminar)",
    "3) Geometría & ORV (con explicación)",
    "4) Álgebra paso a paso (ligado a fórmulas)",
    "5) Resultados",
    "6) Sensibilidad parent–child",
    "PDF",
    "Elemento en foco",
])

# ====== 1) Entradas ======
with tabs[0]:
    st.markdown("#### Definitions & Parameters")
    st.markdown(
        r"""
        Under the conditions assumed in this app, the pressure–transient
        response of a horizontal well with $n_f$ identical transverse hydraulic
        fractures can be modeled by considering one of the fractures producing
        from a rectangular reservoir section at a rate equal to
        $q_f = q/n_f$, where $q$ is the total flow rate of the horizontal well
        """
    )
    st.markdown(
        r"""
        The fracture is located centrally in the closed rectangular drainage area of size
        $2 x_e \times 2 y_e$, which is equal to $1/n_f$ of the total drainage
        area of the horizontal well.  The fracture has a half–length of $x_F$
        and width of $w_F$ and penetrates the entire thickness, $h$, of the
        formation.
        """
    )
    st.markdown("#### Parámetros (Field Units)")
    c1,c2,c3 = st.columns(3)
    with c1:
        mu_cp   = st.number_input(f"μ [{SYMS['mu'].unit}]", 0.01, 50.0, 1.0, 0.01, help=_h("mu"), key="in_mu")
        k_srv_nD= st.number_input(f"k_SRV [{SYMS['k'].unit}]", 0.1, 1e6, 800.0, 0.1, help="Permeabilidad SRV", key="in_k_srv")
    with c2:
        k_orv_nD= st.number_input(f"k_ORV [{SYMS['k'].unit}]", 0.1, 1e6, 150.0, 0.1, help="Permeabilidad ORV", key="in_k_orv")
        ct_invpsi = st.number_input(f"c_t [{SYMS['ct'].unit}]", 1e-7, 1e-2, 1.0e-6, format="%.2e", help=_h("ct"), key="in_ct")
    with c3:
        h_ft     = st.number_input(f"h [{SYMS['h'].unit}]", 1.0, 300.0, 65.6, 0.1, help=_h("h"), key="in_h")
        p0_psi   = st.number_input(f"p_res [{SYMS['p0'].unit}]", 100.0, 10000.0, 4350.0, 1.0, help=_h("p0"), key="in_p0")

# ====== 2) Pozos & schedules: AGREGAR / EDITAR / ELIMINAR ======
if "wells" not in st.session_state:
    st.session_state.wells = [Well(x=0.0, y=0.0, nF=3, xF=50.0, t_days=[0.0], q_stbd=[800.0])]

with tabs[1]:
    st.markdown("#### Alta de pozos")
    cA,cB,cC,cD,cE = st.columns(5)
    with cA: x_new = st.number_input("x nuevo [ft]", -50000.0, 50000.0, 660.0, 10.0, key="add_x")
    with cB: y_new = st.number_input("y nuevo [ft]", -50000.0, 50000.0, 0.0, 10.0, key="add_y")
    with cC: nF_new= st.number_input("n_F nuevo", 1, 50, 3, 1, help=_h("nF"), key="add_nF")
    with cD: xF_new= st.number_input("x_F nuevo [ft]", 1.0, 1000.0, 50.0, 1.0, help=_h("xF"), key="add_xF")
    with cE:
        if st.button("Agregar pozo", key="btn_add"):
            st.session_state.wells.append(Well(x=float(x_new),y=float(y_new),nF=int(nF_new),xF=float(xF_new),
                                               t_days=[0.0], q_stbd=[0.0]))
            st.success("Pozo agregado.")

    st.markdown("---")
    st.markdown("#### Editar / eliminar pozos existentes")
    to_delete=[]
    for i,w in enumerate(st.session_state.wells):
        with st.expander(f"w{i+1} — (x={w.x:.1f}, y={w.y:.1f}) · n_F={w.nF} · x_F={w.xF:.1f}", expanded=False):
            c1,c2,c3,c4 = st.columns(4)
            with c1: x_i = st.number_input(f"x w{i+1} [ft]", -1e6, 1e6, float(w.x), 1.0, key=f"edit_x_{i}")
            with c2: y_i = st.number_input(f"y w{i+1} [ft]", -1e6, 1e6, float(w.y), 1.0, key=f"edit_y_{i}")
            with c3: nF_i= st.number_input(f"n_F w{i+1}", 1, 100, int(w.nF), 1, key=f"edit_nF_{i}")
            with c4: xF_i= st.number_input(f"x_F w{i+1} [ft]", 1.0, 2000.0, float(w.xF), 1.0, key=f"edit_xF_{i}")
            t_line = st.text_input(f"t_k [day] w{i+1}", ", ".join(str(t) for t in w.t_days), key=f"edit_t_{i}")
            q_line = st.text_input(f"q_k [STB/D] w{i+1}", ", ".join(str(q) for q in w.q_stbd), key=f"edit_q_{i}")
            c5,c6 = st.columns(2)
            with c5:
                if st.button(f"Actualizar w{i+1}", key=f"btn_upd_{i}"):
                    try:
                        st.session_state.wells[i].x=float(x_i); st.session_state.wells[i].y=float(y_i)
                        st.session_state.wells[i].nF=int(nF_i); st.session_state.wells[i].xF=float(xF_i)
                        st.session_state.wells[i].t_days=[float(v.strip()) for v in t_line.split(",") if v.strip()]
                        st.session_state.wells[i].q_stbd=[float(v.strip()) for v in q_line.split(",") if v.strip()]
                        ensure_schedule(st.session_state.wells[i]); st.success("Actualizado.")
                    except Exception as e: st.error(f"Error: {e}")
            with c6:
                if st.button(f"Eliminar w{i+1}", key=f"btn_del_{i}"):
                    to_delete.append(i)
    if to_delete:
        for idx in sorted(to_delete, reverse=True): del st.session_state.wells[idx]
        st.warning(f"Eliminados: {', '.join('w'+str(i+1) for i in to_delete)}")

    # mapa planta rápido
    xs=[w.x for w in st.session_state.wells]; ys=[w.y for w in st.session_state.wells]
    fig=go.Figure()
    fig.add_scatter(x=xs,y=ys,mode="markers+text",text=[f"w{j+1}" for j in range(len(xs))],textposition="top center")
    fig.update_xaxes(title="x [ft] (planta)"); fig.update_yaxes(title="y [ft] (planta)"); fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

# ====== 3) Geometría & ORV ======
if "geom_applied" not in st.session_state:
    st.session_state.geom_applied = dict(Lx_srv_ft=200.0, Lx_orv_int_ft=660.0,
                                         Lx_orv_end_ft=2000.0, mode="Lateral explícito (2N−1)")

with tabs[2]:
    Nw = len(st.session_state.wells)
    g_ap = st.session_state.geom_applied  # geometría aplicada vigente
    _Lx_srv_show  = g_ap["Lx_srv_ft"]
    _Lx_orve_show = g_ap["Lx_orv_end_ft"]

    st.markdown("### Geometría y **partición de ORV** — diferencias clave")

    # — Apilado (vertical)
    st.markdown("**Apilado (vertical)**")
    st.markdown("- Con $N$ pozos → **outers = $N+1$**.")
    st.markdown("- Distancias laterales patrón:  $D_{ij}=|i-j|\\,L_{x,\\mathrm{ORV,int}}$.")
    st.markdown("- Impedancia propia (suma de *slabs* SRV + ORV sin flujo):")
    st.latex(r"""
\mathcal R_{ii}(s)=\frac{\mu}{h}\left(
\frac{1}{k_{\mathrm{SRV}}}\frac{\coth(\lambda_{\mathrm{SRV}}\,L_{x,\mathrm{SRV}})}{\lambda_{\mathrm{SRV}}}+
\frac{1}{k_{\mathrm{ORV}}}\frac{\coth(\lambda_{\mathrm{ORV}}\,L_{x,\mathrm{ORV,end}})}{\lambda_{\mathrm{ORV}}}
\right)
""")
    show_eq("Rslab")
    st.markdown(
        chips_row([
            ("μ",          mu_cp,         "cP", 3),
            ("h",          h_ft,          "ft", 3),
            ("k_SRV",      k_srv_nD,      "nD", 3),
            ("k_ORV",      k_orv_nD,      "nD", 3),
            ("Lx_SRV",     _Lx_srv_show,  "ft", 3),
            ("Lx_ORV,end", _Lx_orve_show, "ft", 3),
        ]),
        unsafe_allow_html=True
    )

    # — Lateral explícito (2N−1)
    st.markdown("**Lateral explícito (2N−1)**")
    st.markdown("- Con $N$ pozos → **segmentos ORV = $2N-1$** (entre pozos + 2 extremos).")
    st.markdown("- Distancias reales por posiciones: $D_{ij}=|x_i-x_j|$.")
    st.markdown("- Influencia cruzada (kernel difusivo en ORV):")
    st.latex(r"""
\mathcal R_{ij}(s)=\frac{\mu}{k_{\mathrm{ORV}}\,h}\,
\frac{e^{-\lambda_{\mathrm{ORV}}\,D_{ij}}}{\lambda_{\mathrm{ORV}}},\qquad i\neq j
""")
    show_eq("Rij")

    # Distancias actuales usando SIEMPRE lo aplicado
    if g_ap["mode"].startswith("Apilado"):
        D_ft = distances_apilado(st.session_state.wells, g_ap["Lx_orv_int_ft"])
    else:
        D_ft = distances_lateral(st.session_state.wells)

    pairs=[]
    if len(st.session_state.wells)>=2:
        for i in range(min(3, len(st.session_state.wells)-1)):
            pairs.append((f"D_{i+1}{i+2}", D_ft[i,i+1], "ft", 3))

    st.markdown(
        chips_row([("μ", mu_cp, "cP", 3), ("h", h_ft, "ft", 3), ("k_ORV", k_orv_nD, "nD", 3)] + pairs),
        unsafe_allow_html=True
    )

    # — Definición común
    st.markdown("**Definición común**")
    st.latex(r"""\lambda=\sqrt{\frac{\mu\,c_t}{k}\,s}""")
    show_eq("lambda")

    # Controles (estos sí editan y luego podés "Aplicar geometría")
    c1,c2,c3 = st.columns(3)
    with c1:
        Lx_srv_ft = st.number_input("Lx_SRV [ft]", 1.0, 50000.0, g_ap["Lx_srv_ft"], 1.0, key="geo_Lx_srv")
    with c2:
        Lx_orv_int_ft = st.number_input("Lx_ORV,int [ft]", 1.0, 100000.0, g_ap["Lx_orv_int_ft"], 1.0, key="geo_Lx_orv_int")
    with c3:
        Lx_orv_end_ft = st.number_input("Lx_ORV,end [ft]", 0.0, 200000.0, g_ap["Lx_orv_end_ft"], 1.0, key="geo_Lx_orv_end")
    mode_orv = st.radio("Partición ORV", ["Apilado (vertical)", "Lateral explícito (2N−1)"],
                        index=1 if g_ap["mode"].startswith("Lateral") else 0, key="geo_mode")


    # SVG dinámico
    svg = (svg_orv_apilado(Nw, Lx_srv_ft, Lx_orv_int_ft, Lx_orv_end_ft)
           if mode_orv.startswith("Apilado")
           else svg_orv_lateral(st.session_state.wells, Lx_srv_ft, Lx_orv_end_ft))
    st.components.v1.html(svg, height=380)

    if st.button("Aplicar geometría", key="geo_apply"):
        st.session_state.geom_applied = dict(Lx_srv_ft=float(Lx_srv_ft), Lx_orv_int_ft=float(Lx_orv_int_ft),
                                             Lx_orv_end_ft=float(Lx_orv_end_ft), mode=mode_orv)
        st.success(f"Aplicado: {mode_orv}")

# ====== 4) Álgebra paso a paso (ligado a fórmulas) ======
def build_R_qhat_field_at_s(wells, mu_cp,k_srv_nD,k_orv_nD,ct_invpsi,h_ft,
                            Lx_srv_ft,Lx_orv_end_ft,mode_orv,Lx_orv_int_ft, s_star):
    mu  = si_mu(mu_cp); kSRV=si_k(k_srv_nD); kORV=si_k(k_orv_nD)
    ct  = si_ct(ct_invpsi); h=si_h(h_ft); Ls=si_L(Lx_srv_ft); Lo=si_L(Lx_orv_end_ft)
    D_ft = distances_apilado(wells, Lx_orv_int_ft) if mode_orv=="Apilado (vertical)" else distances_lateral(wells)
    D = si_L(D_ft)
    n=len(wells); R_SI = np.zeros((n,n), float)
    for i in range(n): R_SI[i,i] = R_self(mu, ct, kSRV, kORV, h, Ls, Lo, s_star)
    for i in range(n):
        for j in range(n):
            if i!=j: R_SI[i,j] = R_cross_lateral(mu, ct, kORV, h, D[i,j], s_star)
    R_field = (R_SI/PSI_TO_PA) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)  # [psi·day / STB]
    qhat_STB = np.array([q_hat_piecewise_s(w, s_star) for w in wells], float)  # [STB]
    return R_field, qhat_STB

with tabs[3]:
    # --- Formulación (SPE-215031-PA) → cómo la usa el código  ---
    st.markdown("#### Formulación (SPE-215031-PA) → **cómo la usa el código**")

    # 1) Transformada del caudal por tramos
    # 1) q̂(s) por tramos
    st.markdown("**1) Transformada del caudal por tramos (por pozo)**")
    st.latex(r"\hat q_j(s)=\frac{1}{s}\sum_{k}\big(q_k-q_{k-1}\big)\,e^{-s\,t_{k-1}}")
    show_eq("qhat")        # <<<<<<<<<<
    st.caption("Implementado en `q_hat_piecewise_s`.")

    # 2) Impedancias
    st.markdown("**2) Impedancias** (se usan en `R_self` y `R_cross_lateral`)")
    st.latex(r"\lambda=\sqrt{\frac{\mu\,c_t}{k}\,s}")
    show_eq("lambda")      # <<<<<<<<<<
    st.latex(r"\mathcal R_{\text{slab}}=\frac{\mu}{k\,h}\,\frac{\coth(\lambda L)}{\lambda}")
    show_eq("Rslab")       # <<<<<<<<<<
    st.latex(r"\mathcal R_{\text{semiinf}}=\frac{\mu}{k\,h}\,\frac{1}{\lambda}")
    show_eq("Rsemi")       # <<<<<<<<<<
    st.latex(r"\mathcal R_{ij}(s)=\frac{\mu}{k_{\mathrm{ORV}}\,h}\,\frac{e^{-\lambda_{\mathrm{ORV}}\,D_{ij}}}{\lambda_{\mathrm{ORV}}},\quad i\neq j")
    show_eq("Rij")         # <<<<<<<<<<

    # 3) Sistema matricial
    st.markdown("**3) Sistema matricial en Laplace**")
    st.latex(r"\hat{\mathbf p}_{\mathrm{wf}}(s)=\mathbf R(s)\,\hat{\mathbf q}(s)")
    show_eq("system")      # <<<<<<<<<<


    c1,c2 = st.columns(2)
    with c1: t_star_day = st.number_input("t* [day]", 1e-6, 1e6, 10.0, format="%.6f", key="alg_t_star")
    with c2: k_idx = st.slider("k (nodo Stehfest)", 1, 20, 6, 1, key="alg_k_idx")
    s_star = (k_idx*math.log(2.0))/(t_star_day*DAY_TO_S)
    st.latex(rf"s^*=\dfrac{{{k_idx}\,\ln 2}}{{{t_star_day}\ \mathrm{{day}}}}={s_star:.3e}\ \mathrm{{s^{{-1}}}}")

    # Geometría aplicada primero
    g = st.session_state.geom_applied

    # Sustitución numérica para λ ...
    mu_SI = si_mu(mu_cp); ct_SI = si_ct(ct_invpsi)
    kSRV_SI = si_k(k_srv_nD); kORV_SI = si_k(k_orv_nD)
    h_SI = si_h(h_ft); Ls_SI = si_L(g["Lx_srv_ft"]); Lo_SI = si_L(g["Lx_orv_end_ft"])


    lam_SRV = _lambda(mu_SI, ct_SI, kSRV_SI, s_star)      # [1/m]
    lam_ORV = _lambda(mu_SI, ct_SI, kORV_SI, s_star)      # [1/m]
    lam_SRV_ft = lam_SRV*FT_TO_M                          # [1/ft]
    lam_ORV_ft = lam_ORV*FT_TO_M                          # [1/ft]

    st.markdown("**Valores usados en estas fórmulas (en este $s^*$):**", help="Se actualizan cuando cambiás t*, k o las entradas.")
    st.markdown(
        chips_row([
            ("s*", s_star, "1/s", 3),
            ("λ_SRV", lam_SRV_ft, "1/ft", 3),
            ("λ_ORV", lam_ORV_ft, "1/ft", 3),
            ("Lx_SRV", g["Lx_srv_ft"], "ft", 3),
            ("Lx_ORV,end", g["Lx_orv_end_ft"], "ft", 3),
            ("h", h_ft, "ft", 3),
            ("μ", mu_cp, "cP", 3),
            ("c_t", ct_invpsi, "1/psi", 3),
            ("k_SRV", k_srv_nD, "nD", 3),
            ("k_ORV", k_orv_nD, "nD", 3),
        ]),
        unsafe_allow_html=True
    )
    st.latex(rf"\lambda_{{\mathrm{{SRV}}}}(s^*)=\sqrt{{\frac{{\mu c_t}}{{k_{{\mathrm{{SRV}}}}}}\,s^*}}"
            rf"= {lam_SRV:.3e}\ \mathrm{{m^{{-1}}}} = {lam_SRV_ft:.3e}\ \mathrm{{ft^{{-1}}}}")
    st.latex(rf"\lambda_{{\mathrm{{ORV}}}}(s^*)=\sqrt{{\frac{{\mu c_t}}{{k_{{\mathrm{{ORV}}}}}}\,s^*}}"
            rf"= {lam_ORV:.3e}\ \mathrm{{m^{{-1}}}} = {lam_ORV_ft:.3e}\ \mathrm{{ft^{{-1}}}}")

    g = st.session_state.geom_applied
    R_field, qhat_STB = build_R_qhat_field_at_s(st.session_state.wells, mu_cp,k_srv_nD,k_orv_nD,ct_invpsi,h_ft,
                                                g["Lx_srv_ft"], g["Lx_orv_end_ft"], g["mode"], g["Lx_orv_int_ft"], s_star)
    nW=len(st.session_state.wells)

    st.markdown("**Vector de caudales** $\\hat{\\mathbf q}(s^*)$ [STB]:")
    rows_q = " \\\\ ".join(f"{v:.3e}" for v in qhat_STB)
    st.latex(rf"\hat{{\mathbf q}}(s^*)=\begin{{bmatrix}}{rows_q}\end{{bmatrix}}\ \mathrm{{STB}}")

    st.markdown("**Matriz** $\\mathbf R(s^*)$ [psi·day / STB]:")
    figR = go.Figure(data=go.Heatmap(z=R_field, x=[f"q̂{j+1}" for j in range(nW)], y=[f"p̂{j+1}" for j in range(nW)],
                                     colorscale="Blues"))
    figR.update_layout(height=300); st.plotly_chart(figR, use_container_width=True)

    st.markdown("**Ecuaciones por renglón** $\\hat p_i(s^*)=\\sum_j R_{ij}(s^*)\\hat q_j(s^*)$")
    for i in range(nW):
        terms = " + ".join([rf"{R_field[i,j]:.3e}\,\hat q_{{{j+1}}}" for j in range(nW)])
        st.latex(rf"\hat p_{{{i+1}}}(s^*)={terms}\ [\mathrm{{psi\cdot day}}]")
    p_hat = R_field.dot(qhat_STB)
    rows_p = " \\\\ ".join(f"{v:.3e}" for v in p_hat)
    st.markdown("**Vector** $\\hat{\\mathbf p}(s^*)$ [psi·day]:")
    st.latex(rf"\hat{{\mathbf p}}(s^*)=\begin{{bmatrix}}{rows_p}\end{{bmatrix}}\ \mathrm{{psi\cdot day}}")
    if len(st.session_state.wells)>=2:
        D_now = distances_apilado(st.session_state.wells, g["Lx_orv_int_ft"]) if g["mode"].startswith("Apilado") \
                else distances_lateral(st.session_state.wells)
        # texto compacto: D_12, D_13, ...
        lines=[]
        for i in range(len(st.session_state.wells)):
            for j in range(i+1, len(st.session_state.wells)):
                lines.append(f"D_{{{i+1}{j+1}}}={D_now[i,j]:.2f}\\ \\mathrm{{ft}}")
        st.markdown("**Distancias actuales**:")
        st.latex(",\\ ".join(lines))

# ====== 5) Resultados ======
with tabs[4]:
    st.markdown("#### Pwf(t), ΔP y derivada (usa la geometría aplicada)")
    c1,c2 = st.columns(2)
    with c1: tmin_d = st.number_input("t_min [day]", 1e-6, 1e2, 1e-3, format="%.6f", key="res_tmin")
    with c2: tmax_d = st.number_input("t_max [day]", 1e-4, 1e6, 1e3, format="%.4f", key="res_tmax")
    npts = st.number_input("n_pts", 16, 2000, 220, 1, key="res_npts")
    times_s = np.logspace(np.log10(tmin_d), np.log10(tmax_d), int(npts))*DAY_TO_S

    g = st.session_state.geom_applied
    Fvec = make_Fvec(st.session_state.wells, mu_cp,k_srv_nD,k_orv_nD,ct_invpsi,h_ft,
                     g["Lx_srv_ft"], g["Lx_orv_end_ft"], g["mode"], g["Lx_orv_int_ft"])

    st.session_state.fs_calls_stehfest=0; st.session_state.fs_calls_gaver=0
    nW=len(st.session_state.wells); P=np.zeros((len(times_s), nW)); p_res=p0_psi
    for i,ti in enumerate(times_s):
        pwf = invert_stehfest_vec(Fvec, ti, Nst) if inv_method.startswith("Stehfest") \
               else invert_gaver_euler_vec(Fvec, ti, M=M_gv, P=P_gv)
        P[i,:]= p_res - pwf

    fig=go.Figure()
    for j in range(nW): fig.add_scatter(x=times_s/DAY_TO_S, y=P[:,j], mode="lines", name=f"w{j+1}: Pwf [psi]")
    fig.update_xaxes(title="t [day]", type="log"); fig.update_yaxes(title="Pwf [psi]")
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02)); st.plotly_chart(fig, use_container_width=True)

    if nW>=2:
        dP=P[:,0]-P[:,1]; fig2=go.Figure()
        fig2.add_scatter(x=times_s/DAY_TO_S, y=dP, mode="lines", name="ΔP[w1−w2] [psi]")
        fig2.update_xaxes(title="t [day]", type="log"); fig2.update_yaxes(title="ΔP [psi]"); fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    if nW>=1:
        y=P[:,0]; deriv=np.gradient(np.log(np.maximum(y,1e-12)), np.log(times_s))
        figB=go.Figure(); figB.add_scatter(x=times_s/DAY_TO_S, y=deriv, mode="lines", name="d log P / d log t (w1)")
        figB.update_xaxes(title="t [day]", type="log"); figB.update_yaxes(title="Derivada"); figB.update_layout(height=300)
        st.plotly_chart(figB, use_container_width=True)

    st.caption(f"Llamadas a F(s) — Stehfest: {st.session_state.fs_calls_stehfest} · Gaver(Euler): {st.session_state.fs_calls_gaver}")

# ====== 6) Sensibilidad ======
with tabs[5]:
    st.markdown("#### Barrido x_F del child (pozo 2)")
    t_ref_d = st.number_input("t* [day]", 1e-6, 1e6, 10.0, format="%.6f", key="sens_t_star")
    g = st.session_state.geom_applied
    if len(st.session_state.wells)>=2:
        w1 = st.session_state.wells[0]; w2 = st.session_state.wells[1]
        xF_vals=np.linspace(max(1.0,w2.xF*0.5), w2.xF*1.8, 15); dP_list=[]
        for xF_i in xF_vals:
            w2_tmp=Well(x=w2.x,y=w2.y,nF=w2.nF,xF=xF_i,t_days=w2.t_days,q_stbd=w2.q_stbd)
            wells_tmp=[w1,w2_tmp]+st.session_state.wells[2:]
            Ftmp = make_Fvec(wells_tmp, mu_cp,k_srv_nD,k_orv_nD,ct_invpsi,h_ft,
                             g["Lx_srv_ft"], g["Lx_orv_end_ft"], g["mode"], g["Lx_orv_int_ft"])
            pwf = invert_stehfest_vec(Ftmp, t_ref_d*DAY_TO_S, Nst) if inv_method.startswith("Stehfest") \
                   else invert_gaver_euler_vec(Ftmp, t_ref_d*DAY_TO_S, M=M_gv, P=P_gv)
            P1=p0_psi-pwf[0]; P2=p0_psi-pwf[1]; dP_list.append(P1-P2)
        figS=go.Figure(); figS.add_scatter(x=xF_vals, y=dP_list, mode="lines+markers", name="ΔP(t*) vs x_F (w2)")
        figS.update_xaxes(title="x_F [ft]"); figS.update_yaxes(title="ΔP [psi]"); figS.update_layout(height=320)
        st.plotly_chart(figS, use_container_width=True)
    else:
        st.info("Agregá un segundo pozo (child).")

# ====== 7) PDF ======
with tabs[6]:
    st.subheader("SPE-215031-PA (PDF)")
    raw=None
    if upl is not None: raw=upl.read()
    elif st.session_state.get("pdf_path") and os.path.exists(st.session_state["pdf_path"]):
        try:  raw=open(st.session_state["pdf_path"],"rb").read()
        except Exception: raw=None
    elif default_guess := find_pdf_candidate():
        try:  raw=open(default_guess,"rb").read()
        except Exception: raw=None
    if raw: embed_pdf_bytes(raw, height=720)
    else: st.warning("No pude mostrar el PDF. Dejá el .pdf en la carpeta o usá ‘Browse files’.")

# ====== 8) Elemento en foco ======
with tabs[7]:
    st.markdown("### Elemento en foco — superposición en Laplace e inversión Stehfest")
    st.caption("Elegí i (destino), j (fuente), un tiempo t* y el nodo k. Se muestran R_ij(s*), q_j(s*), el aporte y la descomposición Stehfest.")

    Nw = len(st.session_state.wells)
    if Nw < 1:
        st.info("Agregá al menos un pozo en la pestaña 2.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    with c1: i_sel = c1.number_input("i (destino)", 1, max(1,Nw), 1, 1) - 1
    with c2: j_sel = c2.number_input("j (fuente)", 1, max(1,Nw), min(2,Nw), 1) - 1
    with c3: t_star_day = c3.number_input("t* [day]", 1e-6, 1e6, 10.0, format="%.6f")
    with c4: Nst_focus = c4.slider("N Stehfest", 8, 20, 12, 2)
    t_star = t_star_day*DAY_TO_S
    # nodos y pesos Stehfest
    V = stehfest_weights(Nst_focus)
    s_nodes = (np.arange(1, Nst_focus+1)*math.log(2.0))/max(t_star,1e-30)
    k_idx = st.slider("k (nodo)", 1, Nst_focus, min(6,Nst_focus), 1)
    s_star = s_nodes[k_idx-1]

    # Propiedades actuales desde sesión (usar claves de entradas)
    mu_cp = float(st.session_state.get("in_mu", 1.0))
    k_srv_nD = float(st.session_state.get("in_k_srv", 800.0))
    k_orv_nD = float(st.session_state.get("in_k_orv", 150.0))
    ct_invpsi = float(st.session_state.get("in_ct", 1e-6))
    h_ft = float(st.session_state.get("in_h", 65.6))

    # Geometría aplicada
    g = st.session_state.get("geom_applied", dict(Lx_srv_ft=100.0, Lx_orv_end_ft=10.0, Lx_orv_int_ft=10.0, mode="Apilado (vertical)"))

    # R(s*) y q_hat(s*) usando helper existente
    R_at_s, qhat_s = build_R_qhat_field_at_s(
        st.session_state.wells,
        mu_cp, k_srv_nD, k_orv_nD, ct_invpsi, h_ft,
        g["Lx_srv_ft"], g["Lx_orv_end_ft"], g.get("mode","Apilado (vertical)"), g.get("Lx_orv_int_ft", 10.0),
        s_star
    )
    pihat_s = R_at_s.dot(qhat_s)
    Rij = float(R_at_s[i_sel, j_sel]); qj = float(qhat_s[j_sel]); contrib = Rij*qj

    st.markdown(
        f"s* = {s_star:.3e} 1/s  ·  R_ij(s*) = {Rij:.3e} [psi·day/STB]  ·  q_j(s*) = {qj:.3e} [STB]  ·  "
        f"aporte = R_ij·q_j = {contrib:.3e} [psi·day]  ·  p_i(s*) = {pihat_s[i_sel]:.3e} [psi·day]"
    )

    # Barras: contribuciones Stehfest para p_i(t*)
    pihats = []
    for s_k in s_nodes:
        Rk, qhk = build_R_qhat_field_at_s(
            st.session_state.wells,
            mu_cp, k_srv_nD, k_orv_nD, ct_invpsi, h_ft,
            g["Lx_srv_ft"], g["Lx_orv_end_ft"], g.get("mode","Apilado (vertical)"), g.get("Lx_orv_int_ft", 10.0),
            float(s_k)
        )
        pihats.append( (Rk.dot(qhk))[i_sel] )
    pihats = np.array(pihats, float)
    terms = (math.log(2.0)/max(t_star,1e-30)) * V * pihats
    figB = go.Figure(data=[go.Bar(x=[f"k={k}" for k in range(1, Nst_focus+1)], y=terms)])
    figB.update_layout(height=260, title="Descomposición Stehfest de p_i(t*) (términos por nodo)", yaxis_title="[psi]")
    st.plotly_chart(figB, use_container_width=True)

    # Kernel cruzado vs distancia (para s*)
    # Distancias actuales según modo
    D_ft = distances_apilado(st.session_state.wells, g.get("Lx_orv_int_ft", 10.0)) if g.get("mode","Apilado (vertical)")=="Apilado (vertical)" \
           else distances_lateral(st.session_state.wells)
    Ds_ft = np.linspace(0.0, max(1.0, np.max(D_ft)*1.05), 80)
    mu = si_mu(mu_cp); ct = si_ct(ct_invpsi); kO = si_k(k_orv_nD); h = si_h(h_ft)
    Ds_SI = si_L(Ds_ft)
    lamO = _lambda(mu, ct, kO, s_star)
    Rcross_vsD_SI = (mu/(kO*max(h,1e-12))) * np.exp(-lamO*Ds_SI) / max(lamO,1e-30)
    # Convertir a [psi·day / STB]
    Rcross_vsD = (Rcross_vsD_SI/PSI_TO_PA) * DAY_TO_S / (STB_TO_M3/DAY_TO_S)
    figK = go.Figure()
    figK.add_scatter(x=Ds_ft, y=Rcross_vsD, mode="lines", name="R_cross(s*, D)")
    figK.add_vline(x=float(D_ft[i_sel,j_sel]) if D_ft.size>0 else 0.0, line=dict(dash="dash"))
    figK.update_layout(height=260, title="Kernel cruzado R_ij(s*, D) vs D  (línea punteada: D_ij actual)",
                       xaxis_title="D [ft]", yaxis_title="R [psi·day/STB]")
    st.plotly_chart(figK, use_container_width=True)
