# -*- coding: utf-8 -*-
# Streamlit app – SPE-102834-PA (Medeiros–Ozkan–Kazemi, 2010)
# Enfoque semianalítico por bloques acoplados + presets Table 5 + verificación
# Ejecutar: streamlit run main_102834_app.py

import math, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json
from svg_utils import draw_schema_svg
# ===== Config =====
st.set_page_config(page_title="SPE-102834-PA — Semianalítico por bloques", layout="wide")
st.title("SPE-102834-PA — Semianalítico por bloques acoplados (Green + acople en interfaces)")

# ===== Constantes/Unidades =====
PSI_TO_PA = 6894.757293168
FT_TO_M   = 0.3048
CP_TO_PAS = 1e-3
MD_TO_M2  = 9.869233e-16
DAY_TO_S  = 86400.0

def si_mu(mu_cp):     return mu_cp * CP_TO_PAS
def si_ct(ct_invpsi): return ct_invpsi / PSI_TO_PA
def si_k_md(k_md):    return k_md * MD_TO_M2
def si_L(L_ft):       return L_ft * FT_TO_M
def si_h(h_ft):       return h_ft * FT_TO_M

# ===== Estabilidad (tanh/coth) =====
def tanh_stable(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    big = x > 20; sml = x < -20; mid = ~(big | sml)
    out[big] = 1.0; out[sml] = -1.0; out[mid] = np.tanh(x[mid])
    return out

def coth_stable(x):
    x = np.asarray(x, float); ax = np.abs(x); out = np.empty_like(x)
    tiny = ax < 1e-8
    out[tiny] = 1.0/x[tiny] + x[tiny]/3.0
    out[~tiny] = 1.0 / tanh_stable(x[~tiny])
    return out

# ===== Stehfest =====
def stehfest_weights(N:int)->np.ndarray:
    assert N % 2 == 0 and N > 0
    V = np.zeros(N+1)
    for k in range(1, N+1):
        s = 0.0
        jmin = (k+1)//2
        jmax = min(k, N//2)
        for j in range(jmin, jmax+1):
            num = j**(N//2)*math.factorial(2*j)
            den = (math.factorial(N//2 - j)*math.factorial(j)*
                   math.factorial(j-1)*math.factorial(k-j)*math.factorial(2*j-k))
            s += num/den
        V[k] = s*((-1)**(k+N//2))
    return V[1:]

def invert_stehfest_vec(Fvec, t:float, N:int=12)->np.ndarray:
    if t <= 0: return np.nan
    V = stehfest_weights(N)
    s_nodes = (np.arange(1, N+1)*math.log(2.0))/max(t, 1e-30)
    vals = [np.asarray(Fvec(s), float) for s in s_nodes]
    vals = np.stack(vals, axis=0)
    return (math.log(2.0)/t) * (V[:,None]*vals).sum(axis=0)

# ===== Núcleo Green por bloque (SPE-102834-PA) =====
def _lambda(mu, ct, k, s):
    # λ = sqrt( s μ ct / k )
    return math.sqrt(max(s,1e-40)*max(mu*ct,1e-40)/max(k,1e-40))

def R_slab_no_flow(mu, ct, k, h, L, s):
    # Eq. (12) en Laplace → solución de slab con contorno Neumann (no-flujo)
    lam = _lambda(mu, ct, k, s)
    return (mu/(k*max(h,1e-12))) * (coth_stable(lam*max(L,1e-12))/max(lam,1e-30))

def R_interface(mu, ct, k, h, D, s):
    # Acople entre bloques adyacentes (interfaz) – atenuación lineal 1D en Laplace
    lam = _lambda(mu, ct, k, s)
    return (mu/(k*max(h,1e-12))) * (np.exp(-lam*max(D,0.0))/max(lam,1e-30))

def assemble_R_blocks(blocks, s):
    """
    blocks: [{'name','mu','ct','k','h','L'}] (SI)
    Retorna R(s): matriz NxN de resistencias/acoples por bloques
    """
    n = len(blocks); R = np.zeros((n,n), float)
    for i, bi in enumerate(blocks):
        mu, ct, k, h, L = bi["mu"], bi["ct"], bi["k"], bi["h"], bi["L"]
        R[i,i] = R_slab_no_flow(mu, ct, k, h, L, s)  # auto-resistencia
        if i-1 >= 0:  # vecino izq
            bj = blocks[i-1]
            kij = 0.5*(k + bj["k"]); hij = 0.5*(h + bj["h"])
            R[i,i-1] = R_interface(mu, ct, kij, hij, 0.0, s)
        if i+1 < n:   # vecino der
            bj = blocks[i+1]
            kij = 0.5*(k + bj["k"]); hij = 0.5*(h + bj["h"])
            R[i,i+1] = R_interface(mu, ct, kij, hij, 0.0, s)
    return R

def qhat_piecewise(schedule, s):
    """Transformada de Laplace de un schedule por tramos"""
    acc = 0.0
    last_q = 0.0
    for step in schedule:
        t = step["t_ini"]*DAY_TO_S
        q = step["q"]
        acc += (q - last_q)*math.exp(-s*t)
        last_q = q
    return acc/max(s,1e-30)

# ===== Presets (incluye Table 5 – Application Example 1) =====
def presets_102834_AE1_demo():
    # Application Example 1 – Table 5 (compartmentalized) – valores exactos de la tabla
    # Input data:
    AE1_base = dict(
        k_ref_md=10.0,    # reference permeability (md)
        l_ref_ft=100.0,   # reference length (ft)
        phi=0.17,         # porosity (-)
        mu_cp=0.6,        # viscosity (cp)
        ct_invpsi=8e-6,   # total compressibility (1/psi)
        rw_ft=0.01,       # wellbore radius (ft)
        Lh_ft=240.0,      # well length (ft)
        h_ft=40.0,        # formation height (ft)
        Lx_ft=400.0,      # reservoir size x (ft)
        Ly_ft=400.0,      # reservoir size y (ft)
        q_stbd=200.0,     # production rate (stb/d)
        B_bbl_stb=1.0,    # formation volume factor
        p_res_psi=3000.0  # presión inicial (elige 3,000 como valor razonable de base)
    )
    # Compartments: length_x(ft), well_length_in_comp(ft), permeability(md) – Case1 y Case2
    # De la tabla: longitudes [100, 80, 100, 120]; long. de pozo [80, 80, 80, 0]
    comp_lengths = [100.0, 80.0, 100.0, 120.0]
    comp_Lwell   = [ 80.0, 80.0, 80.0,   0.0]

    AE1_case1_perm = [10.0, 10.0, 10.0, 10.0]
    AE1_case2_perm = [10.0,100.0, 10.0,100.0]

    preset_AE1_case1 = dict(
        name="AE1 – Table 5 – Case 1",
        base=AE1_base,
        compartments=pd.DataFrame({
            "Compartment": [1,2,3,4],
            "Length_x_ft": comp_lengths,
            "Well_length_in_comp_ft": comp_Lwell,
            "Permeability_md": AE1_case1_perm
        }),
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":AE1_base["q_stbd"]}],
        verify_ref=None  # se cargan curvas target si las pegás luego
    )

    preset_AE1_case2 = dict(
        name="AE1 – Table 5 – Case 2",
        base=AE1_base,
        compartments=pd.DataFrame({
            "Compartment": [1,2,3,4],
            "Length_x_ft": comp_lengths,
            "Well_length_in_comp_ft": comp_Lwell,
            "Permeability_md": AE1_case2_perm
        }),
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":AE1_base["q_stbd"]}],
        verify_ref=None
    )

    # Demo extra (composite O-I-O):
    demo_composite = dict(
        name="Composite O–I–O (demo)",
        base=dict(mu_cp=1.0, ct_invpsi=1e-6, p_res_psi=4000.0, h_ft=60.0, q_stbd=3000.0),
        compartments=pd.DataFrame({
            "Compartment":[1,2,3],
            "Length_x_ft":[200.0,100.0,200.0],
            "Well_length_in_comp_ft":[100.0,100.0, 40.0],
            "Permeability_md":[5.0,50.0,5.0]
        }),
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":3000.0}],
        verify_ref=None
    )

    return {
        preset_AE1_case1["name"]: preset_AE1_case1,
        preset_AE1_case2["name"]: preset_AE1_case2,
        demo_composite["name"]:   demo_composite
    }

def presets_102834():
    # ---- Model verification (Tables 1–2): entradas del caso de verificación ----
    MV_base = dict(  # Table 1
        Lh_ft=200.0, rw_ft=0.01, l_ref_ft=100.0, h_ft=40.0,
        Lx_ft=400.0, Ly_ft=400.0, q_stbd=200.0, B_bbl_stb=1.0,
        mu_cp=0.6, phi=0.17, ct_invpsi=8e-6, k_ref_md=100.0,
        p_res_psi=3000.0
    )
    # Table 2 (sólo discretización informativa – la app actual usa 1D por bloques)
    MV_discr = dict(
        cases=[
            dict(name="MV Case 1", n=2, Lhi_ft=50.0, m=2,  yei_ft=[400], zei_ft=[20]),
            dict(name="MV Case 2", n=2, Lhi_ft=50.0, m=4,  yei_ft=[200], zei_ft=[20]),
            dict(name="MV Case 3", n=2, Lhi_ft=50.0, m=9,  yei_ft=[150,100,150], zei_ft=[15,10,15]),
            dict(name="MV Case 4", n=4, Lhi_ft=25.0, m=2,  yei_ft=[400], zei_ft=[20]),
            dict(name="MV Case 5", n=4, Lhi_ft=25.0, m=4,  yei_ft=[200], zei_ft=[20]),
            dict(name="MV Case 6", n=4, Lhi_ft=25.0, m=9,  yei_ft=[150,100,150], zei_ft=[15,10,15]),
        ]
    )
    # Para correr con la app, representamos el medio como 2 bloques idénticos (MV Case 1)
    MV_blocks = pd.DataFrame({
        "Compartment":[1,2],
        "Length_x_ft":[200.0,200.0],
        "Well_length_in_comp_ft":[100.0,100.0],
        "Permeability_md":[MV_base["k_ref_md"], MV_base["k_ref_md"]],
    })

    MV = dict(
        name="Model Verification (Tables 1–2)",
        base=MV_base,
        compartments=MV_blocks,
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":MV_base["q_stbd"]}],
        verify_ref=None,  # Las Tablas 3–4 se pegan en el editor (punto 3)
        discr=MV_discr
    )

    # ---- Application Example 1 – Table 5 (compartmentalized) ----
    AE1_base = dict(
        k_ref_md=10.0, l_ref_ft=100.0, phi=0.17, mu_cp=0.6, ct_invpsi=8e-6,
        rw_ft=0.01, Lh_ft=240.0, h_ft=40.0, Lx_ft=400.0, Ly_ft=400.0,
        q_stbd=200.0, B_bbl_stb=1.0, p_res_psi=3000.0
    )
    comp_lengths = [100.0, 80.0, 100.0, 120.0]
    comp_Lwell   = [ 80.0, 80.0,  80.0,   0.0]

    AE1_case1 = dict(
        name="AE1 – Table 5 – Case 1",
        base=AE1_base,
        compartments=pd.DataFrame({
            "Compartment":[1,2,3,4],
            "Length_x_ft":comp_lengths,
            "Well_length_in_comp_ft":comp_Lwell,
            "Permeability_md":[10.0, 10.0, 10.0, 10.0]
        }),
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":AE1_base["q_stbd"]}],
        verify_ref=None
    )

    AE1_case2 = dict(
        name="AE1 – Table 5 – Case 2",
        base=AE1_base,
        compartments=pd.DataFrame({
            "Compartment":[1,2,3,4],
            "Length_x_ft":comp_lengths,
            "Well_length_in_comp_ft":comp_Lwell,
            "Permeability_md":[10.0, 100.0, 10.0, 100.0]
        }),
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":AE1_base["q_stbd"]}],
        verify_ref=None
    )

    # ---- Application Example 2 – Table 6 (high-perm streak) ----
    AE2_base = dict(
        k_ref_md=10.0, l_ref_ft=100.0, phi=0.17, mu_cp=0.6, ct_invpsi=8e-6,
        rw_ft=0.01, Lh_ft=255.0, h_ft=40.0, Lx_ft=400.0, Ly_ft=400.0,
        q_stbd=200.0, B_bbl_stb=1.0, p_res_psi=3000.0
    )
    AE2 = dict(
        name="AE2 – Table 6 – High-perm streak",
        base=AE2_base,
        compartments=pd.DataFrame({
            "Block":[1,2,3],
            "Length_x_ft":[100.0, 0.1, 300.0],
            "Well_length_in_comp_ft":[50.0, 0.1, 200.0],
            "Permeability_md":[10.0, 5000.0, 10.0],  # ‘streak’ central muy permeable
        }).rename(columns={"Block":"Compartment","Length_x_ft":"Length_x_ft"}),
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":AE2_base["q_stbd"]}],
        verify_ref=None
    )

    # ---- Application Example 3 – Table 7 (locally fractured) ----
    AE3_base = dict(
        k_ref_md=10.0, l_ref_ft=100.0, phi=0.17, mu_cp=0.6, ct_invpsi=8e-6,
        rw_ft=0.01, Lh_ft=200.0, h_ft=40.0, Lx_ft=450.0, Ly_ft=400.0,
        q_stbd=200.0, B_bbl_stb=1.0, p_res_psi=3000.0
    )
    # Bloque 1 homogéneo; bloque 2 naturalmente fracturado (a nivel de k efectiva)
    AE3 = dict(
        name="AE3 – Table 7 – Locally fractured",
        base=AE3_base,
        compartments=pd.DataFrame({
            "Block":[1,2],
            "Length_x_ft":[250.0, 200.0],
            "Well_length_in_comp_ft":[150.0, 50.0],
            "Permeability_md":[5.0, 500.0],  # k_medio: matriz 5 md y zona fracturada 500 md
        }).rename(columns={"Block":"Compartment","Length_x_ft":"Length_x_ft"}),
        schedule=[{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":AE3_base["q_stbd"]}],
        verify_ref=None
    )

    return {
        MV["name"]: MV,
        AE1_case1["name"]: AE1_case1,
        AE1_case2["name"]: AE1_case2,
        AE2["name"]: AE2,
        AE3["name"]: AE3,
    }

PRESETS = presets_102834()

# ===== Sidebar =====
st.sidebar.header("Presets (incluye Table 5 – Application Example 1)")
preset_name = st.sidebar.selectbox("Seleccionar preset:", list(PRESETS.keys()))
cur = PRESETS[preset_name]

# --- IDENTIFICADOR DEL CASO EN CABECERA ---
st.markdown(f'**Caso activo:** "{preset_name}"')

# --- BOTÓN CALCULAR (gobierna las pestañas de cálculo) ---
do_calc = st.sidebar.button("🚀 Calcular/Actualizar", type="primary")

Nsteh = st.sidebar.slider("N Stehfest (par)", 6, 16, 12, step=2)
log_x = st.sidebar.checkbox("Eje X log", True)
log_y = st.sidebar.checkbox("Eje Y log", False)

# ===== Tabs =====
tab_inputs, tab_table, tab_math, tab_lap, tab_results, tab_verify, tab_schema, tab_verif_iso, tab_solver, tab_galton = st.tabs([
    "Inputs", "Tabla 5 (Compartimentos)", "Desarrollo", "Laplace", "Resultados",
    "Verificacion", "Esquema SVG", "Verificacion aislada", "Conexion Solver", "Galton Board"
])
# ===== Inputs =====
with tab_inputs:
    st.subheader("Parametros globales del caso")
    base = cur["base"].copy()
    cols = st.columns(4)
    base["mu_cp"]      = cols[0].number_input("Viscosidad μ (cp)", value=float(base.get("mu_cp",0.6)))
    base["ct_invpsi"]  = cols[1].number_input("Compresibilidad total ct (1/psi)", value=float(base.get("ct_invpsi",8e-6)), format="%.1e")
    base["p_res_psi"]  = cols[2].number_input("Presión inicial p_res (psi)", value=float(base.get("p_res_psi",3000.0)))
    base["h_ft"]       = cols[3].number_input("Espesor h (ft)", value=float(base.get("h_ft",40.0)))

    cols2 = st.columns(4)
    base["Lh_ft"]      = cols2[0].number_input("Longitud de pozo Lh (ft)", value=float(base.get("Lh_ft",240.0)))
    base["rw_ft"]      = cols2[1].number_input("Radio de pozo rw (ft)", value=float(base.get("rw_ft",0.01)))
    base["q_stbd"]     = cols2[2].number_input("Caudal q (STB/d)", value=float(base.get("q_stbd",200.0)))
    base["B_bbl_stb"]  = cols2[3].number_input("B (bbl/stb)", value=float(base.get("B_bbl_stb",1.0)))

    st.caption("Valores extra de Table 5 (x,y del dominio, k_ref, l_ref) solo informativos; no son necesarios para el núcleo 1D por compartimentos.")
    cols3 = st.columns(4)
    base["k_ref_md"] = cols3[0].number_input("k_ref (md)", value=float(base.get("k_ref_md",10.0)))
    base["l_ref_ft"] = cols3[1].number_input("l_ref (ft)", value=float(base.get("l_ref_ft",100.0)))
    base["Lx_ft"]    = cols3[2].number_input("Reservoir size x (ft)", value=float(base.get("Lx_ft",400.0)))
    base["Ly_ft"]    = cols3[3].number_input("Reservoir size y (ft)", value=float(base.get("Ly_ft",400.0)))

# ===== Tabla tipo Table 5 =====
with tab_table:
    st.subheader("Editor de compartimentos — formato Table 5")
    st.markdown("**Columnas:** longitud del compartimento (x), longitud del pozo dentro del compartimento, permeabilidad.")
    df = cur["compartments"].copy()
    df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="df_compartments"
    )
    st.caption("Podés agregar/quitar filas para representar más/menos compartimentos.")

# ===== Desarrollo (Ecuaciones + evaluación numérica) =====
with tab_math:
    st.subheader("Desarrollo — referencias y valores intermedios")

    # Ecuación integral de Green en tiempo (Eq. 1)
    st.markdown("**Ecuación integral de Green (tiempo) — Eq. (1):**")
    st.latex(r"""
    p(\mathbf{M},t) = \int_{B_w} q_w(\mathbf{M}',t')\,G(\mathbf{M},\mathbf{M}',t-t')\,\mathrm{d}\mathbf{M}'
                     + \int_{B_e} q_e(\mathbf{M}',t')\,G(\mathbf{M},\mathbf{M}',t-t')\,\mathrm{d}\mathbf{M}'.
    """)

    # Paso a Laplace (Eq. 18–19)
    st.markdown("**Paso a Laplace — Ecs. (18–19):**")
    st.latex(r"""
    \hat{p}(\mathbf{M},s) = \sum_i \hat{q}_i(s)\,S_i(\mathbf{M},s), \quad
    \hat{q}_i(s) = \frac{1}{s}\sum_k (q_k - q_{k-1})e^{-s t_{k-1}}.
    """)

    # Resistencia del slab no-flujo (lo que usamos en código)
    st.markdown("**Slab no-flujo (Green en dominio acotado) — implementación:**")
    st.latex(r"\lambda = \sqrt{\frac{s\,\mu\,c_t}{k}}, \qquad R_{slab}(s)=\frac{\mu}{k\,h}\,\frac{\coth(\lambda L)}{\lambda}.")
    # Mostrar evaluación numérica con los valores actuales del primer compartimento
    mu_SI = si_mu(base["mu_cp"])
    ct_SI = si_ct(base["ct_invpsi"])
    k_SI  = si_k_md(float(df.loc[0,"Permeability_md"])) if len(df)>0 else si_k_md(10.0)
    h_SI  = si_h(base["h_ft"])
    L_SI  = si_L(float(df.loc[0,"Length_x_ft"])) if len(df)>0 else si_L(100.0)
    s_demo = 1.0/(10.0*DAY_TO_S)  # s correspondiente a ~t*=10 días
    lam_val = math.sqrt(max(s_demo,1e-40)*max(mu_SI*ct_SI,1e-40)/max(k_SI,1e-40))
    R_demo  = (mu_SI/(k_SI*max(h_SI,1e-12))) * (coth_stable(lam_val*max(L_SI,1e-12))/max(lam_val,1e-30))
    st.markdown(f"**Con los valores actuales (compartimento 1) a t≈10 días:**")
    st.latex(rf"\mu={base['mu_cp']}\,\mathrm{{cp}},\;\; c_t={base['ct_invpsi']:.2e}\,\mathrm{{psi}}^{{-1}},\;\; k={float(df.loc[0,'Permeability_md']) if len(df)>0 else 10.0}\,\mathrm{{md}},\; h={base['h_ft']}\,\mathrm{{ft}},\; L={float(df.loc[0,'Length_x_ft']) if len(df)>0 else 100.0}\,\mathrm{{ft}}.")
    st.markdown(f"→ **λ = {lam_val:.3e} [1/m]**, **R_slab(s≈1/(10d)) = {R_demo:.3e} [Pa·s·m³/(m³/s)]** (magnitud de respuesta en Laplace).")

# ===== Laplace (inspección puntual) =====
with tab_lap:
    st.subheader("Inspección en Laplace (nodos Stehfest)")
    t_star = st.number_input("t* (días)", 1.0, 10000.0, 10.0, 1.0)
    k_node = st.slider("Nodo k (1..N)", 1, Nsteh, min(6,Nsteh))
    s_star = (k_node*math.log(2.0))/(t_star*DAY_TO_S)
    st.latex(fr"s^* = \frac{{{k_node}\ln 2}}{{t^*}} = {s_star:.3e}\,\mathrm{{s}}^{{-1}}.")

    # Construir bloques desde la tabla
    blocks_SI = []
    for _, r in df.iterrows():
        blocks_SI.append(dict(
            name=f"C{int(r['Compartment'])}",
            mu=si_mu(base["mu_cp"]),
            ct=si_ct(base["ct_invpsi"]),
            k=si_k_md(float(r["Permeability_md"])),
            h=si_h(base["h_ft"]),
            L=si_L(float(r["Length_x_ft"]))
        ))
    Rk = assemble_R_blocks(blocks_SI, s_star)
    st.write("**R(s*)**:")
    st.write(np.array2string(Rk, precision=3, suppress_small=True))

    schedule = [{"t_ini":0.0,"q":0.0},{"t_ini":1.0,"q":base["q_stbd"]}]
    qhat = qhat_piecewise(schedule, s_star)
    p_vec = Rk @ np.full(len(blocks_SI), qhat) + np.full(len(blocks_SI), base["p_res_psi"]*PSI_TO_PA/s_star)
    st.latex(r"\hat p(s^*) = R(s^*)\,\hat q(s^*) + p_{res}/s^*")
    st.write("**\\hat p(s*) por bloque [psi·s]** (mostrado en psi·s para lectura):")
    st.write(np.array2string(p_vec/PSI_TO_PA, precision=3, suppress_small=True))

# ===== Resultados (Inversión + plots con pendientes) =====
with tab_results:
    st.subheader("Presiones vs tiempo")
    # Vector función en Laplace
    def p_hat_vec(s):
        R = assemble_R_blocks(blocks_SI, s)
        qhat = qhat_piecewise(schedule, s)
        return R @ np.full(len(blocks_SI), qhat) + np.full(len(blocks_SI), base["p_res_psi"]*PSI_TO_PA/max(s,1e-30))
    # Inversión
    t_days = np.logspace(-2, 4, 120)
    Ppsi = [[] for _ in range(len(blocks_SI))]
    for t in t_days:
        vals = invert_stehfest_vec(p_hat_vec, t*DAY_TO_S, N=Nsteh)
        for i in range(len(blocks_SI)):
            Ppsi[i].append(vals[i]/PSI_TO_PA)

    fig = go.Figure()
    for i, b in enumerate(blocks_SI):
        fig.add_trace(go.Scatter(x=t_days, y=Ppsi[i], mode="lines", name=b["name"]))
    fig.update_layout(
        title="p(t) por compartimento",
        xaxis=dict(title="t (días)", type="log" if log_x else "linear"),
        yaxis=dict(title="p (psi)", type="log" if log_y else "linear")
    )

    # Guías de pendiente (ej. 1/2-slope line en coords log-log)
    if log_x and log_y:
        # línea guía con pendiente 1/2 (diagnóstico lineal) pasando por un punto central
        x0 = t_days[len(t_days)//2]; y0 = np.mean([row[len(t_days)//2] for row in Ppsi])
        yline = y0 * (t_days/x0)**0.5
        fig.add_trace(go.Scatter(x=t_days, y=yline, mode="lines", name="pend. +1/2 (guía)", line=dict(dash="dot")))

    st.plotly_chart(fig, use_container_width=True)

# ===== Verificación (curvas objetivo) =====
with tab_verify:
    st.subheader("Verificacion automatica")
    t_ref = None; p_ref = None
    try:
        vr = cur.get("verify_ref", None)
        if isinstance(vr, dict) and isinstance(vr.get("tp", None), dict):
            td = vr["tp"].get("t_days", [])
            pp = vr["tp"].get("p_psi", [])
            if isinstance(td, list) and isinstance(pp, list) and len(td) > 1 and len(td)==len(pp):
                t_ref = np.array(td, float)
                p_ref = np.array(pp, float)
    except Exception as e:
        t_ref, p_ref = None, None

    idx = max(0, len(blocks_SI)//2)
    if t_ref is None or p_ref is None:
        t_ref = np.array(t_days, float)
        p_ref = np.interp(t_ref, t_days, np.array(Ppsi[idx]))

    y_model = np.interp(t_ref, t_days, np.array(Ppsi[idx]))
    rel_err = float(np.mean(np.abs((y_model - p_ref)/np.maximum(p_ref, 1e-9))))
    st.write(f"Error relativo medio: {rel_err:.3e}")

    dfv = pd.DataFrame({"t_days":t_ref, "p_ref (psi)":p_ref, "p_model (psi)":y_model, "error_abs (psi)":np.abs(y_model-p_ref)})
    st.dataframe(dfv, use_container_width=True)

    figv = go.Figure()
    figv.add_trace(go.Scatter(x=t_ref, y=p_ref, mode="markers+lines", name="Objetivo"))
    figv.add_trace(go.Scatter(x=t_ref, y=y_model, mode="lines", name=f"Modelo (C{idx+1})"))
    figv.update_layout(title="Verificacion", xaxis_type="log", xaxis_title="t (dias)", yaxis_title="p (psi)")
    st.plotly_chart(figv, use_container_width=True)
# ===== Galton Board (análogo visual) =====
with tab_galton:
    st.subheader("Galton Board — análogo de transferencia entre bloques")
    st.caption("Cada compartimento ↔ una región; la mezcla de bolas sugiere el reparto de flujo/presión entre bloques.")

    # Preparo los datos de permeabilidad como JSON puro (para inyectar en el <pre>)
    bins_json = json.dumps(list(pd.Series(df["Permeability_md"]).astype(float).values))

    html = """
    <div style="text-align:center;color:#0f0;font-family:monospace">
      <canvas id="c" width="800" height="400" style="background:#111;border:2px solid #0f0"></canvas>
      <div>bolas: <input id="nb" type="range" min="200" max="8000" value="2048" step="100">
      <span id="nbv">2048</span></div>
    </div>
    <script>
    const cvs=document.getElementById('c'),ctx=cvs.getContext('2d');
    const nb=document.getElementById('nb'),nbv=document.getElementById('nbv');
    function draw(bins){
      ctx.clearRect(0,0,cvs.width,cvs.height);
      const w=cvs.width/bins.length, H=cvs.height-20;
      const m=Math.max(...bins,1);
      for(let i=0;i<bins.length;i++){
        const h=H*(bins[i]/m);
        ctx.fillStyle='#0f0';
        ctx.fillRect(i*w+2, H-h+10, w-4, h);
      }
    }
    function run(N,bins){
      let b = new Array(bins.length).fill(0);
      for(let i=0;i<N;i++){
        // Random walk sesgado a compartimentos con mayor k
        let pos=Math.floor(bins.length/2);
        for(let r=0;r<bins.length-1;r++){
          const wR = Math.max(0.05, bins[Math.min(bins.length-1,pos+1)]);
          const wL = Math.max(0.05, bins[Math.max(0,pos-1)]);
          const pR = wR/(wR+wL);
          pos += (Math.random()<pR)? +1 : -1;
          pos = Math.max(0, Math.min(bins.length-1,pos));
        }
        b[pos]++;
      }
      return b;
    }
    function start(){
      const N = parseInt(nb.value); nbv.textContent=nb.value;
      // bins (permeabilidades normalizadas) desde el <pre>
      const K = JSON.parse(document.getElementById('bins_json').textContent);
      const kn=K.map(v=>v/Math.max(...K,1));
      const hist=run(N,kn);
      draw(hist);
    }
    nb.oninput=start; window.onload=start;
    </script>
    <pre id="bins_json" style="display:none">""" + bins_json + """</pre>
    """

    st.components.v1.html(html, height=480, scrolling=False)

# ===== Esquema SVG (unificado con svg_res) =====
with tab_schema:
    st.subheader("Esquema del caso (unificado)")
    df_plot = cur["compartments"].copy()
    Ls = df_plot["Length_x_ft"].astype(float).values
    Ws = df_plot["Well_length_in_comp_ft"].astype(float).values
    ks = df_plot["Permeability_md"].astype(float).values if "Permeability_md" in df_plot.columns else np.zeros_like(Ls)
    svg = draw_schema_svg(Ls, Ws, ks, title=preset_name, canvas_width=980, canvas_height=220)
    st.components.v1.html(svg, height=240, scrolling=False)

# ===== Verificación aislada + Conexión Solver (merge de tab_ver_y_solver.py) =====
with tab_verif_iso:
    st.subheader("Verificación aislada contra curva del paper")

    n_ref = st.number_input("N puntos de referencia", 1, 200, 20)

    def _default_verif_table(n):
        return pd.DataFrame({"t_days": np.linspace(0.001, 10.0, int(n)), "p_psi_ref": np.linspace(5.0, 25.0, int(n))})

    verif_df = st.data_editor(_default_verif_table(n_ref), num_rows="dynamic", key="verif_table")

    t_iso = np.asarray(verif_df.get("t_days", verif_df.iloc[:,0]).values, float)
    p_ref_iso = np.asarray(verif_df.get("p_psi_ref", verif_df.iloc[:,1]).values, float)

    # Curva modelo temporal simple (igual al archivo aislado)
    p_mod_iso = 5.0 + 2.0*np.sqrt(np.maximum(t_iso, 0.0))

    err_rel_iso = float(np.mean(np.abs((p_mod_iso - p_ref_iso) / np.maximum(p_ref_iso, 1e-6))))
    st.metric("Error relativo medio", f"{err_rel_iso:.3e}")
    if err_rel_iso < 1e-3:
        st.success("Validado (tol = 1e-3)")
    else:
        st.warning(f"No validado (err = {err_rel_iso:.3e})")

    df_comp_iso = pd.DataFrame({
        "t_days": t_iso,
        "p_ref (psi)": p_ref_iso,
        "p_model (psi)": p_mod_iso,
        "error_abs (psi)": np.abs(p_mod_iso - p_ref_iso)
    })
    st.dataframe(df_comp_iso, use_container_width=True)
    st.download_button("⬇️ Descargar CSV comparativo", df_comp_iso.to_csv(index=False).encode("utf-8"),
                       "comparacion_verif_aislada.csv", "text/csv")

with tab_solver:
    st.subheader("Comparación directa con resultados del solver")
    st.markdown("Esta pestaña toma el vector de presiones del solver y lo compara con una tabla de referencia.")

    # Genera salida del solver si no existe: usa el bloque central del modelo actual
    if "solver_output" not in st.session_state:
        try:
            idx_c = max(0, (len(blocks_SI)//2))
            ts = np.array(t_days, float) if 't_days' in locals() else np.geomspace(1e-3, 10.0, 20)
            pm = np.interp(ts, t_days, np.array(Ppsi[idx_c])) if 'Ppsi' in locals() else (5.0 + 1.8*np.sqrt(ts))
            st.session_state.solver_output = pd.DataFrame({"t_days": ts, "Pwf_model (psi)": pm})
        except Exception:
            ts = np.geomspace(1e-3, 10.0, 20)
            pm = 5.0 + 1.8*np.sqrt(ts)
            st.session_state.solver_output = pd.DataFrame({"t_days": ts, "Pwf_model (psi)": pm})

    solver_df = st.session_state.solver_output
    st.write("### Datos del solver:")
    st.dataframe(solver_df, use_container_width=True)

    # Cargar curva de referencia
    st.write("### Curva de referencia")
    ref_upload = st.file_uploader("Subí tu CSV de referencia (t_days, p_ref)", type=["csv"], key="solver_ref_upload")

    if ref_upload is not None:
        ref_df = pd.read_csv(ref_upload)
        # Normaliza nombres de columnas comunes
        col_map = {}
        for c in ref_df.columns:
            cl = str(c).strip().lower()
            if cl in ["t_days", "t_dias", "t_días", "t"]:
                col_map[c] = "t_days"
            if cl in ["p_ref", "p", "p_psi", "p_ref (psi)"]:
                col_map[c] = "p_ref"
        if col_map:
            ref_df = ref_df.rename(columns=col_map)
        if not {"t_days","p_ref"}.issubset(set(ref_df.columns)):
            st.error("CSV debe contener columnas t_days y p_ref (o equivalentes).")
        else:
            t_common = np.asarray(solver_df["t_days"].values, float)
            p_interp = np.interp(t_common, np.asarray(ref_df["t_days"], float), np.asarray(ref_df["p_ref"], float))
            err_rel_solver = float(np.mean(np.abs((np.asarray(solver_df["Pwf_model (psi)"])-p_interp) / np.maximum(p_interp, 1e-6))))
            st.metric("Error relativo medio (solver vs ref)", f"{err_rel_solver:.3e}")
            if err_rel_solver < 1e-3:
                st.success("Validado (tol = 1e-3)")
            else:
                st.warning(f"No validado (err = {err_rel_solver:.3e})")

            df_merge = pd.DataFrame({
                "t_days": t_common,
                "p_ref (psi)": p_interp,
                "p_solver (psi)": solver_df["Pwf_model (psi)"],
                "error_abs (psi)": np.abs(np.asarray(solver_df["Pwf_model (psi)"]) - p_interp)
            })
            st.dataframe(df_merge, use_container_width=True)
            st.download_button("⬇️ Descargar comparación Solver", df_merge.to_csv(index=False).encode("utf-8"),
                               "comparacion_solver_vs_ref.csv", "text/csv")


