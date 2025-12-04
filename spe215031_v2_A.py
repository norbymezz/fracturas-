# ─────────────────────────────────────────────────────────────────────────────
# SPE-215031-PA • Interferencia de pozos horizontales (app desde cero, v2)
# PARTE A: base + ejemplos + geometría (SVG) + doble porosidad (default ON)
# Ejecutá: streamlit run spe215031_v2_A.py
# ─────────────────────────────────────────────────────────────────────────────

import math, os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Iterable

import numpy as np
import streamlit as st

# ======================
# 0) Config & estilo UI
# ======================
st.set_page_config(page_title="SPE-215031-PA — Interferencia (v2)", layout="wide")
st.title("SPE-215031-PA — Interferencia de pozos horizontales (v2)")

st.markdown("""
<style>
.block-container { padding-top: 0.8rem; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { padding: 10px 14px; }
.stButton>button { padding: .45rem .8rem; }
.dataframe tbody tr td, .dataframe thead th { padding: .22rem .4rem; font-size: .88rem; }
.small-help { color:#9fb6d9; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

# ==========================
# 1) Glosario + LaTeX únicos
# ==========================
COLORS: Dict[str, str] = {
    "bg": "#0b1220", "text": "#cfe8ff",
    "srv": "#2d6cdf", "orv": "#f4d03f", "orv_end": "#32b5ff",
    "dist": "#8b1a1a", "len": "#111111", "total": "#27ae60",
    "well": "#3b3b3b", "edge": "#000000", "tick": "#999999"
}

GLOSS = {
    "N": ("Número de pozos paralelos.", "Dimensiona matrices y gaps."),
    "L_well": ("Longitud horizontal de los pozos.", "Solo visual; no entra en R(s)."),
    "Lx_I": ("Semi-longitud SRV (⊥ al pozo).", "En R_self con k_I."),
    "2x_Oi": ("Ancho total ORV extremo (2·x_O,i).", "En R_self con k_O (ORV_end)."),
    "spacing": ("Distancia entre ejes i–i+1.", "Define Lx_O,int_g y D_ij."),
    "nf": ("Número de fracturas por pozo.", "Solo visual (ticks equiespaciados)."),
    "xf": ("Semi-longitud de fractura.", "Informativa."),
    "omega": ("ω (fracc. almacenamiento en fracturas).", "Factor DP f_K(s) (Eq. 30)."),
    "Lambda": ("Λ (transmisividad inter-poros).", "Factor DP f_K(s) (Eq. 30)."),
    "kI": ("k_I (permeab. SRV).", "En λ_I y R_self."),
    "kO": ("k_O (permeab. ORV).", "En λ_O, R_self y R_ij."),
    "mu": ("μ (viscosidad).", "En todos los kernels."),
    "ct": ("c_t (compresibilidad total).", "En λ."),
    "h": ("h (espesor).", "En todos los kernels."),
    "p0": ("p_res (presión inicial).", "Para Pwf(t)=p_res−p̂^{-1}."),
}
def help_of(k): a,b=GLOSS.get(k,("","")); return f"{a}\n**En el código:** {b}" if b else a

LATEX = {
    "lambda":  r"\lambda_K=\sqrt{\frac{\mu\,c_t}{k_K}\,s}",
    "Rslab":   r"\mathcal R_{\text{slab}}=\frac{\mu}{k\,h}\,\frac{\coth(\lambda L)}{\lambda}",
    "Rsemi":   r"\mathcal R_{\text{semiinf}}=\frac{\mu}{k\,h}\,\frac{1}{\lambda}",
    "Rij":     r"\mathcal R_{ij}(s)=\frac{\mu}{k_{\mathrm{O}}\,h}\,\frac{e^{-\lambda_{\mathrm{O}}\,D_{ij}}}{\lambda_{\mathrm{O}}},\ i\neq j",
    "qhat":    r"\hat q_j(s)=\frac{1}{s}\sum_{\ell}(q_\ell-q_{\ell-1})\,e^{-s\,t_{\ell-1}}",
    "system":  r"\hat{\mathbf p}(s)=\mathbf R(s)\,\hat{\mathbf q}(s)",
    "fK":      r"f_K(s)=\omega_K + \sqrt{\frac{\Lambda_K(1-\omega_K)}{3s}}\,\tanh\!\Big(\sqrt{\frac{3(1-\omega_K)s}{\Lambda_K}}\Big)",
}
def latex_of(k): return LATEX.get(k,"")

# ==========================
# 2) Unidades + funciones núcleo
# ==========================
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

def tanh_stable(x):
    x=np.asarray(x,float); out=np.empty_like(x)
    big=x>20; sml=x<-20; mid=~(big|sml)
    out[big]=1.0; out[sml]=-1.0; out[mid]=np.tanh(x[mid]); return out

def _lambda(mu, ct, k, s):  return math.sqrt(max(s,1e-40)*max(mu*ct,1e-40)/max(k,1e-40))

def fK_dp(s: float, omega: float, Lambda: float) -> float:
    """Eq. (30) — factor de doble porosidad para slab K (I u O)."""
    omega=float(omega); Lambda=float(max(Lambda,1e-30))
    term = math.sqrt(Lambda*(1.0-omega)/max(3.0*s,1e-30))
    arg  = math.sqrt(max(3.0*(1.0-omega)*s/Lambda,0.0))
    return omega + term*float(tanh_stable(arg))

def coth_stable(x):
    x=np.asarray(x,float); ax=np.abs(x); out=np.empty_like(x)
    tiny=ax<1e-8; out[tiny]=1.0/x[tiny]+x[tiny]/3.0; out[~tiny]=1.0/tanh_stable(x[~tiny]); return out

def R_slab_no_flow(mu, ct, k, h, L, s, mult=1.0):
    lam=_lambda(mu,ct,k,s); x=lam*max(L,1e-12)
    base=(mu/(k*max(h,1e-12)))*(coth_stable(x)/max(lam,1e-30))
    return mult*base

def R_semi_inf(mu, ct, k, h, s):
    lam=_lambda(mu,ct,k,s); return (mu/(k*max(h,1e-12)))*(1.0/max(lam,1e-30))

def R_self(mu, ct, k_srv, k_orv, h, Lx_srv, Lx_orv_end, s,
           dp_on_I=True, omega_I=0.3, Lambda_I=1.0,
           dp_on_O=True, omega_O=0.3, Lambda_O=1.0):
    mI = fK_dp(s,omega_I,Lambda_I) if dp_on_I else 1.0
    mO = fK_dp(s,omega_O,Lambda_O) if dp_on_O else 1.0
    R_srv = R_slab_no_flow(mu,ct,k_srv,h,Lx_srv,s,mult=mI)
    R_orv = R_slab_no_flow(mu,ct,k_orv,h,Lx_orv_end,s,mult=mO) if Lx_orv_end>0 else R_semi_inf(mu,ct,k_orv,h,s)
    return R_srv + R_orv

def R_cross_lateral(mu, ct, k_orv, h, Dij, s):
    lam=_lambda(mu,ct,k_orv,s)
    return (mu/(k_orv*max(h,1e-12)))*math.exp(-lam*max(Dij,0.0))/max(lam,1e-30)

# ==========================
# 3) Schedules (tolerante dict/obj)
# ==========================
@dataclass
class WellSched:
    t_days: List[float]
    q_stbd: List[float]

def _as_lists_sched(ws: Union[WellSched, dict, Tuple[Iterable[float], Iterable[float]]]) -> Tuple[List[float], List[float]]:
    if isinstance(ws, WellSched): return list(ws.t_days), list(ws.q_stbd)
    if isinstance(ws, dict):      return list(ws.get("t_days",[])), list(ws.get("q_stbd",[]))
    t,q = ws; return list(t), list(q)

def ensure_schedule(ws: Union[WellSched, dict, Tuple[Iterable[float], Iterable[float]]]) -> WellSched:
    t_list,q_list=_as_lists_sched(ws)
    if not t_list: t_list=[0.0]; q_list=[0.0]
    pairs=sorted(zip(t_list,q_list), key=lambda z: float(z[0]))
    t=[float(pairs[0][0])]; q=[float(pairs[0][1])]
    for ti,qi in pairs[1:]:
        ti=float(ti); qi=float(qi)
        if ti!=t[-1]: t.append(ti); q.append(qi)
        else: q[-1]=qi
    return WellSched(t_days=t,q_stbd=q)

def q_hat_piecewise_s(ws: Union[WellSched, dict], s: float) -> float:
    t_list,q_list = _as_lists_sched(ensure_schedule(ws))
    acc=0.0; last_q=0.0
    for tk_day,qk in zip(t_list,q_list):
        tk = float(tk_day)*DAY_TO_S
        acc += (float(qk)-last_q)*math.exp(-s*tk)
        last_q=float(qk)
    return acc/max(s,1e-30)

# ==========================
# 4) Geometría → números
# ==========================
def build_geometry_numbers(spacing_g_ft: List[float], Lx_I_i_ft: List[float], two_xO_i_ft: List[float]) -> Dict[str, List[float]]:
    """Devuelve Lx_O,int_g, Lx_O,end_i y total (apilado/lateral equivalentes en planta)."""
    N=len(Lx_I_i_ft); gaps=len(spacing_g_ft)
    # ORV interno: medio gap a cada lado ⇒ semi-ancho por gap
    Lx_O_int_g = [sg/2.0 for sg in spacing_g_ft]
    # ORV final por pozo i: tomamos la mitad del 2·x_O,i que cargaste (semi-ancho)
    Lx_O_end_i = [max(0.0, float(v)/2.0) for v in two_xO_i_ft]
    # total = ORV_end(izq) + ∑(SRV_i + ORV_int_g alrededor) + ORV_end(der)
    total = 0.0
    total += Lx_O_end_i[0] + Lx_O_end_i[-1]
    total += sum([2*Lx_I_i_ft[i] for i in range(N)]) * 0.0  # (orientativo, no suma al ancho)
    total += sum(spacing_g_ft)
    return dict(Lx_O_int_g_ft=Lx_O_int_g, Lx_O_end_i_ft=Lx_O_end_i, total_ft=sum(spacing_g_ft)+Lx_O_end_i[0]+Lx_O_end_i[-1])

def distances_lateral_from_spacings(spacing_g_ft: List[float]) -> np.ndarray:
    """D_ij en planta: distancia acumulada de gaps."""
    N = len(spacing_g_ft)+1
    x=[0.0]
    for g in spacing_g_ft: x.append(x[-1]+g)
    X=np.array(x)
    D=np.abs(X.reshape(-1,1)-X.reshape(1,-1))
    return D

# ==========================
# 5) SVG de esquema (fracturas equiespaciadas)
# ==========================
def _svg_header(W,H):
    return [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="{COLORS["bg"]}" />']

def _text(x,y,txt,fs=12,fill=None,anchor="start"):
    fill = fill or COLORS["text"]
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{fs}" text-anchor="{anchor}">{txt}</text>'

def _arrow(x1,y1,x2,y2,color,th=2):
    shaft=f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{th}" />'
    def head(xa,ya,xb,yb):
        ang=math.atan2(yb-ya, xb-xa); L=8; W=5
        xL=xb-L*math.cos(ang); yL=yb-L*math.sin(ang)
        xN1=xL+W*math.cos(ang+math.pi/2); yN1=yL+W*math.sin(ang+math.pi/2)
        xN2=xL+W*math.cos(ang-math.pi/2); yN2=yL+W*math.sin(ang-math.pi/2)
        return f'<polygon points="{xb},{yb} {xN1},{yN1} {xN2},{yN2}" fill="{color}"/>'
    return shaft + head(x2,y2,x1,y1) + head(x1,y1,x2,y2)

def svg_geometry_parallel(
    L_well_ft: float,
    Lx_I_i_ft: List[float],
    two_xO_i_ft: List[float],
    spacing_g_ft: List[float],
    Lx_O_int_g_ft: List[float],
    Lx_O_end_i_ft: List[float],
    total_ft: float,
    nf_list: List[int],
    max_width_px: int = 980,
) -> str:
    N=len(Lx_I_i_ft); gaps=len(spacing_g_ft)
    W=max_width_px; row_h=72; top=60; H=top+N*row_h+140
    x_left=140; x_right=W-150; x_mid=(x_left+x_right)//2
    svg=_svg_header(W,H)
    svg.append(_text(16,28,"Esquema (no a escala) — SRV / ORV / fracturas / cotas"))

    bank_x0=x_mid-26; bank_x1=x_mid+26
    svg.append(f'<rect x="{bank_x0}" y="{top-20}" width="{bank_x1-bank_x0}" height="{N*row_h}" fill="#ececec" stroke="{COLORS["edge"]}" stroke-width="1"/>')

    for r in range(N):
        yc=top+r*row_h+row_h//2
        svg.append(f'<rect x="{x_mid-5}" y="{yc-22}" width="10" height="44" fill="{COLORS["well"]}" />')
        svg.append(_text(x_left-100,yc+5,f"pozo i={r+1}",12,COLORS["text"]))
        # fracturas equiespaciadas
        nf=max(1,int(nf_list[r])) if r<len(nf_list) else 1
        y0=yc-18; y1=yc+18
        for k in range(nf):
            t=(k+0.5)/nf; yk=y0*(1-t)+y1*t
            svg.append(f'<line x1="{x_mid-12}" y1="{yk}" x2="{x_mid+12}" y2="{yk}" stroke="#666" stroke-width="2"/>')
        # SRV
        svg.append(_arrow(x_mid-10,yc,x_left+30,yc,COLORS["srv"],2))
        svg.append(_text((x_mid-10+x_left+30)//2,yc-10,f"2·Lx_I = {2*Lx_I_i_ft[r]:.0f} ft",12,COLORS["srv"],"middle"))
        # ORV final
        svg.append(_arrow(x_mid+10,yc,x_right-30,yc,COLORS["orv_end"],2))
        svg.append(_text((x_mid+10+x_right-30)//2,yc-10,f"2·x_O,i = {two_xO_i_ft[r]:.0f} ft",12,COLORS["orv_end"],"middle"))

    # long. de pozos (única)
    y_up=top+6; y_dn=top+N*row_h-6
    svg.append(_arrow(x_right+28,y_up,x_right+28,y_dn,COLORS["len"],2))
    svg.append(_text(x_right+34,(y_up+y_dn)//2,f"longitud de los pozos y reservorio ≈ {L_well_ft:.0f} ft",12,COLORS["len"]))

    # distancias y ORV interno
    y0=top+row_h//2
    for g in range(gaps):
        ya=y0+g*row_h; yb=ya+row_h
        svg.append(_arrow(x_right-70,ya,x_right-70,yb,COLORS["dist"],2))
        svg.append(_text(x_right-64,(ya+yb)//2,f"dist i={g+1}–i={g+2}: {spacing_g_ft[g]:.0f} ft",12,COLORS["dist"]))
        svg.append(_arrow(x_left-70,ya,x_left-70,yb,COLORS["orv"],2))
        svg.append(_text(x_left-64,(ya+yb)//2,f"Lx_O,int_{g+1} = {Lx_O_int_g_ft[g]:.0f} ft",12,COLORS["orv"]))

    # total
    svg.append(_arrow(x_right-108,top-10,x_right-108,top+N*row_h+10,COLORS["total"],3))
    svg.append(_text(x_right-100,top+N*row_h+30,f"total = {total_ft:.0f} ft",13,COLORS["total"],"start"))

    # leyenda
    legend_y=H-40; lx=22
    def leg(x,c,lab):
        svg.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+24}" y2="{legend_y}" stroke="{c}" stroke-width="6"/>')
        svg.append(_text(x+32,legend_y+5,lab,12))
    leg(lx,COLORS["srv"],"SRV (2·Lx_I)"); lx+=160
    leg(lx,COLORS["orv_end"],"ORV final (2·x_O,i)"); lx+=210
    leg(lx,COLORS["orv"],"ORV interno (Lx_O,int_g)"); lx+=200
    leg(lx,COLORS["dist"],"distancias entre pozos"); lx+=230
    leg(lx,COLORS["total"],"total")
    svg.append("</svg>")
    return "\n".join(svg)

# ==========================
# 6) Datos de ejemplo (precargados)
# ==========================
@dataclass
class WellGeom:
    nf: int = 25
    xf: float = 200.0
    sched: WellSched = WellSched([0.0],[1000.0])

@dataclass
class AppState:
    N: int
    L_well_ft: float
    Lx_I_i_ft: List[float]
    two_xO_i_ft: List[float]
    spacing_g_ft: List[float]
    wells: List[WellGeom]

EXAMPLES: Dict[str, AppState] = {
    # Ejemplo “interferencia FUERTE”: gaps chicos y k_ORV alto (lo verás en ΔP)
    "Interferencia FUERTE (4 pozos)": AppState(
        N=4, L_well_ft=5000.0,
        Lx_I_i_ft=[100,100,100,100],
        two_xO_i_ft=[20,60,100,40],
        spacing_g_ft=[400,420,400],
        wells=[WellGeom(25,200.0,WellSched([0.0],[1000.0])) for _ in range(4)]
    ),
    # Caso más “suave”
    "Interferencia SUAVE (3 pozos)": AppState(
        N=3, L_well_ft=5000.0,
        Lx_I_i_ft=[150,150,150], two_xO_i_ft=[20,20,20],
        spacing_g_ft=[800,900],
        wells=[WellGeom(25,200.0,WellSched([0.0],[800.0])) for _ in range(3)]
    ),
}

# ==========================
# 7) Estado de la app (inicial)
# ==========================
if "app" not in st.session_state:
    init = EXAMPLES["Interferencia FUERTE (4 pozos)"]
    st.session_state.app = init

if "phys" not in st.session_state:
    st.session_state.phys = dict(mu_cp=1.0, kI_nD=800.0, kO_nD=150.0,
                                 ct_invpsi=1.0e-6, h_ft=65.6, p0_psi=4350.0,
                                 dp_on_I=True, omega_I=0.30, Lambda_I=1.0,
                                 dp_on_O=True, omega_O=0.30, Lambda_O=1.0)

# ==========================
# 8) Tabs
# ==========================
tabs = st.tabs([
    "1) Guía & ejemplos",
    "2) Geometría (SVG) & edición",
    "3) Álgebra (con referencias)",
    "4) Matriz R(s*) & resultados",   # implementado en PARTE B
])

# ── 1) Guía & ejemplos ──────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Cómo usar esta versión")
    st.markdown("""
1. **Elegí un ejemplo** para precargar geometría y schedules.  
2. Revisá/ajustá **parámetros de fluido/roca** (doble porosidad viene **activa**).  
3. En **Geometría (SVG)** verificás las **cotas** (SRV, ORV, distancias, total) y fracturas **equiespaciadas**.  
4. En **Álgebra** está la derivación con **LaTeX** y debajo de cada ecuación te indico **cómo se implementa**.  
5. En la pestaña **Matriz R(s*) & resultados** se calcula \\(\\mathbf R(s)\\), se arma \\(\\hat q(s)\\) y se invierte para **Pwf(t)** y **ΔP(t)**.

> Tip: el ejemplo **Interferencia FUERTE** usa **gaps pequeños** → verás ΔP grandes.
    """)
    c1,c2 = st.columns([1.1,1])
    with c1:
        name = st.selectbox("Elegí un escenario:", list(EXAMPLES.keys()))
        if st.button("Cargar ejemplo", use_container_width=True):
            st.session_state.app = EXAMPLES[name]
            st.success(f"Ejemplo cargado: {name}")

    with c2:
        st.markdown("#### Fluido/Roca (Field Units)")
        phys = st.session_state.phys
        mu_cp   = st.number_input("μ [cP]", 0.01, 50.0, phys["mu_cp"], 0.01, help=help_of("mu"), key="mu_cp")
        kI_nD   = st.number_input("k_I [nD]", 0.1, 1e7, phys["kI_nD"], 0.1, help=help_of("kI"), key="kI_nD")
        kO_nD   = st.number_input("k_O [nD]", 0.1, 1e7, phys["kO_nD"], 0.1, help=help_of("kO"), key="kO_nD")
        ct_inv  = st.number_input("c_t [1/psi]", 1e-7, 1e-2, phys["ct_invpsi"], format="%.2e", help=help_of("ct"), key="ct_invpsi")
        h_ft    = st.number_input("h [ft]", 1.0, 300.0, phys["h_ft"], 0.1, help=help_of("h"), key="h_ft")
        p0_psi  = st.number_input("p_res [psi]", 100.0, 10000.0, phys["p0_psi"], 1.0, help=help_of("p0"), key="p0_psi")
        st.session_state.phys.update(dict(mu_cp=mu_cp,kI_nD=kI_nD,kO_nD=kO_nD,ct_invpsi=ct_inv,h_ft=h_ft,p0_psi=p0_psi))
        st.markdown("#### Doble porosidad (default ON)")
        cI,cO = st.columns(2)
        with cI:
            dp_on_I = st.checkbox("DP en I (SRV)", True, help="Eq. 30 aplicado al slab SRV.", key="dp_on_I")
            omega_I = st.slider("ω_I", 0.0, 0.99, 0.30, 0.01, help=help_of("omega"), key="omega_I")
            Lambda_I= st.number_input("Λ_I", 1e-6, 1e6, 1.0, format="%.3g", help=help_of("Lambda"), key="Lambda_I")
        with cO:
            dp_on_O = st.checkbox("DP en O (ORV)", True, help="Eq. 30 aplicado al slab ORV_end.", key="dp_on_O")
            omega_O = st.slider("ω_O", 0.0, 0.99, 0.30, 0.01, help=help_of("omega"), key="omega_O")
            Lambda_O= st.number_input("Λ_O", 1e-6, 1e6, 1.0, format="%.3g", help=help_of("Lambda"), key="Lambda_O")
        st.session_state.phys.update(dict(dp_on_I=dp_on_I,omega_I=omega_I,Lambda_I=Lambda_I,
                                          dp_on_O=dp_on_O,omega_O=omega_O,Lambda_O=Lambda_O))

# ── 2) Geometría (SVG) & edición ────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Editá geometría y schedules. El SVG refleja SRV/ORV, **distancias** y **total**; fracturas se muestran **equiespaciadas**.")
    app: AppState = st.session_state.app

    cA,cB = st.columns([1.05,1.4])
    with cA:
        app.N = st.number_input("N pozos", 2, 24, app.N, 1, help=help_of("N"))
        app.L_well_ft = st.number_input("Longitud de los pozos [ft]", 100.0, 20000.0, app.L_well_ft, 10.0, help=help_of("L_well"))
        st.write("**SRV (Lx_I) y ORV final (2·x_O,i)**")
        if len(app.Lx_I_i_ft)!=app.N: app.Lx_I_i_ft = (app.Lx_I_i_ft+[app.Lx_I_i_ft[-1]])[:app.N]
        if len(app.two_xO_i_ft)!=app.N: app.two_xO_i_ft = (app.two_xO_i_ft+[app.two_xO_i_ft[-1]])[:app.N]
        for i in range(app.N):
            c1,c2 = st.columns(2)
            app.Lx_I_i_ft[i]   = c1.number_input(f"Lx_I (i={i+1}) [ft]", 1.0, 50000.0, float(app.Lx_I_i_ft[i]), 1.0, help=help_of("Lx_I"), key=f"LxI_{i}")
            app.two_xO_i_ft[i] = c2.number_input(f"2·x_O,i (i={i+1}) [ft]", 0.0, 200000.0, float(app.two_xO_i_ft[i]), 1.0, help=help_of("2x_Oi"), key=f"twoXOi_{i}")

        st.write("**Distancias entre pozos**")
        if len(app.spacing_g_ft)!=app.N-1:
            if app.N-1>0:
                base = app.spacing_g_ft[0] if app.spacing_g_ft else 600.0
                app.spacing_g_ft=[base]*(app.N-1)
            else:
                app.spacing_g_ft=[]
        for g in range(max(0,app.N-1)):
            app.spacing_g_ft[g] = st.number_input(f"spacing_{g+1} (i={g+1}–i={g+2}) [ft]", 1.0, 200000.0, float(app.spacing_g_ft[g]), 1.0, help=help_of("spacing"), key=f"sp_{g}")

    with cB:
        st.write("**Pozos & Schedules (rápido)** — nf (fracturas), xf (ft), schedule (t[q]) por pozo.")
        for i in range(app.N):
            if i>=len(app.wells): app.wells.append(WellGeom())
            w=app.wells[i]
            c1,c2,c3 = st.columns([1,1,2])
            w.nf = c1.number_input(f"n_f (i={i+1})", 1, 300, int(w.nf), 1, help=help_of("nf"), key=f"nf_{i}")
            w.xf = c2.number_input(f"x_f (i={i+1}) [ft]", 1.0, 3000.0, float(w.xf), 1.0, help=help_of("xf"), key=f"xf_{i}")
            t_str = ", ".join(str(v) for v in w.sched.t_days); q_str = ", ".join(str(v) for v in w.sched.q_stbd)
            line = c3.text_input(f"schedule i={i+1} (t[q])", f"{t_str} | {q_str}",
                                 help="Formato:  t0, t1, ... | q0, q1, ...   (en day y STB/D)", key=f"sch_{i}")
            try:
                tpart,qpart = line.split("|")
                t_vals=[float(x.strip()) for x in tpart.split(",") if x.strip()!=""]
                q_vals=[float(x.strip()) for x in qpart.split(",") if x.strip()!=""]
                w.sched = ensure_schedule(WellSched(t_vals,q_vals))
            except Exception:
                st.caption(f":red[Formato inválido → se mantiene el anterior para i={i+1}]")

    # números derivados + SVG
    nums = build_geometry_numbers(app.spacing_g_ft, app.Lx_I_i_ft, app.two_xO_i_ft)
    nf_list = [w.nf for w in app.wells[:app.N]]
    svg = svg_geometry_parallel(app.L_well_ft, app.Lx_I_i_ft, app.two_xO_i_ft, app.spacing_g_ft,
                                nums["Lx_O_int_g_ft"], nums["Lx_O_end_i_ft"], nums["total_ft"], nf_list)
    st.components.v1.html(svg, height=380+72*app.N, scrolling=False)

# ── 3) Álgebra (con referencias) ────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Formulación con doble porosidad (activada) y dónde está en el código")
    st.markdown("**1) Transformada del caudal por tramos**")
    st.latex(latex_of("qhat"))
    st.caption("Implementado en `q_hat_piecewise_s(ws, s)`; los schedules se normalizan con `ensure_schedule`.")

    st.markdown("**2) Núcleos básicos y λ**")
    st.latex(latex_of("lambda"))
    st.caption("Función `_lambda(mu, ct, k, s)`.")
    st.latex(latex_of("Rslab"))
    st.caption("`R_slab_no_flow(mu, ct, k, h, L, s, mult)` — el multiplicador `mult` permite la doble porosidad.")
    st.latex(latex_of("Rsemi"))
    st.caption("`R_semi_inf(mu, ct, k, h, s)`.")

    st.markdown("**3) Doble porosidad (Eq. 30)**")
    st.latex(latex_of("fK"))
    st.caption("`fK_dp(s, ω, Λ)`; aplicado en `R_self(..., mult=mI/mO)` con switches `dp_on_I`, `dp_on_O`.")

    st.markdown("**4) Influencia cruzada (en ORV)**")
    st.latex(latex_of("Rij"))
    st.caption("`R_cross_lateral(mu, ct, k_O, h, D_ij, s)`; las distancias se obtienen de los `spacing_g_ft`.")

    st.markdown("**5) Sistema en Laplace**")
    st.latex(latex_of("system"))
    st.caption("En la pestaña siguiente se arma `R(s*)` con tus parámetros y se muestra la multiplicación `R(s*)·q̂(s*)` paso a paso.")

# ── 4) Matriz R(s*) & resultados ────────────────────────────────────────────
# Implementado en archivo B (import dinámico abajo)
try:
    import spe215031_v2_B as partB
    partB.render_results_tab(st, st.session_state,  tabs[3],
                             build_geometry_numbers, distances_lateral_from_spacings,
                             R_self, R_cross_lateral, q_hat_piecewise_s,
                             si_mu, si_k, si_ct, si_h, si_L, field_p,
                             DAY_TO_S, STB_TO_M3)
except Exception as e:
    with tabs[3]:
        st.warning("Para ver esta pestaña, guardá también el archivo **spe215031_v2_B.py** en la misma carpeta.")
        st.caption(f"Detalle técnico: {e}")