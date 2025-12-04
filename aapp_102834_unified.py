# app_102834_unified.py
# Ejecutar:  streamlit run /mnt/data/app_102834_unified.py
# Requisitos: streamlit, numpy, pandas, plotly
# Descripción:
#   App unificada que combina:
#     - Geometría numérica por pozos (SRV/ORV) + matrices R(s) + inversión Laplace (Stehfest y Gaver–Euler)
#     - Editor de pozos & schedules, cotas y esquema (inspirado en aapwm.py)
#     - Pestañas "Álgebra paso a paso", heatmap de R, q̂(s), p̂(s) (inspirado en aaapp.py & HTML)
#     - Loader de 4 "Application Examples" de 102834-PA (tablas y figuras del paper)
#   Incluye un panel de EVALUACIÓN con un criterio pedagógico/funcional auto-calculado.

import math, json
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============== Config UI ==============
st.set_page_config(page_title="SPE-102834-PA — Unified App (Examples 1–4)",
                   layout="wide", initial_sidebar_state="expanded")
st.title("SPE-102834-PA — Unified App (Examples 1–4)")

# ============== Constantes de conversión (Field↔SI) ==============
PSI_TO_PA=6894.757293168; FT_TO_M=0.3048; CP_TO_PAS=1e-3; ND_TO_M2=9.869233e-22
DAY_TO_S=86400.0; STB_TO_M3=0.158987294928
def si_mu(mu_cp): return mu_cp*CP_TO_PAS
def si_ct(ct_invpsi): return ct_invpsi/PSI_TO_PA
def si_h(h_ft): return h_ft*FT_TO_M
def si_k(k_nD): return k_nD*ND_TO_M2
def si_L(L_ft): return L_ft*FT_TO_M

# ============== Estabilidad numérica (tanh/coth/exp) ==============
def tanh_stable(x):
    x=np.asarray(x,float); out=np.empty_like(x)
    big=x>20.0; sml=x<-20.0; mid=~(big|sml)
    out[big]=1.0; out[sml]=-1.0; out[mid]=np.tanh(x[mid]); return out
def coth_stable(x):
    x=np.asarray(x,float); ax=np.abs(x); out=np.empty_like(x)
    tiny=ax<1e-8; out[tiny]=1.0/x[tiny]+x[tiny]/3.0; out[~tiny]=1.0/tanh_stable(x[~tiny]); return out
def exp_clamped(z, lim=700.0): return np.exp(np.clip(z, -lim, lim))

# ============== Laplace (Stehfest / Gaver–Euler) ==============
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

def invert_stehfest_vec(Fvec: Callable[[float], np.ndarray], t: float, N:int)->np.ndarray:
    if t<=0: return np.nan*np.ones_like(Fvec(1.0))
    V=stehfest_weights(N)
    s_nodes=(np.arange(1,N+1)*math.log(2.0))/max(t,1e-30)
    vals=[np.asarray(Fvec(s), float) for s in s_nodes]
    vals=np.stack(vals, axis=0)
    return (math.log(2.0)/t) * (V[:,None]*vals).sum(axis=0)

def invert_gaver_euler_vec(Fvec: Callable[[float], np.ndarray], t: float, M:int=18, P:int=8)->np.ndarray:
    ln2=math.log(2.0); S=[]
    for n in range(1, M+P+1):
        acc=None
        for k in range(1, n+1):
            s=k*ln2/max(t,1e-30)
            term=(((-1)**k)*math.comb(n,k))*np.asarray(Fvec(s),float)
            acc=term if acc is None else acc+term
        S.append(acc)
    S=np.stack(S, axis=0)
    E = sum(math.comb(P,p)*S[(M-1)+p] for p in range(0,P+1))/(2.0**P)
    return (-ln2/max(t,1e-30)) * E

# ============== Núcleo físico tipo SRV/ORV (proxy pedagógico) ==============
def _lambda(mu, ct, k, s):  return math.sqrt(max(s,1e-40)*max(mu*ct,1e-40)/max(k,1e-40))

def R_slab_no_flow(mu, ct, k, h, L, s):
    lam=_lambda(mu,ct,k,s); x=lam*max(L,1e-12)
    return (mu/(k*max(h,1e-12))) * (coth_stable(x)/max(lam,1e-30))

def R_semi_inf(mu, ct, k, h, s):
    lam=_lambda(mu,ct,k,s)
    return (mu/(k*max(h,1e-12))) * (1.0/max(lam,1e-30))

def R_self(mu, ct, k_I, k_O, h, Lx_I, Lx_O_end, s):
    R_I = R_slab_no_flow(mu, ct, k_I, h, Lx_I, s)
    R_O = R_slab_no_flow(mu, ct, k_O, h, Lx_O_end, s) if Lx_O_end>0 else R_semi_inf(mu, ct, k_O, h, s)
    return R_I + R_O

def R_cross(mu, ct, k_O, h, Dij, s):
    lam_O = _lambda(mu, ct, k_O, s)
    return (mu/(k_O*max(h,1e-12))) * float(exp_clamped(-lam_O*max(Dij,0.0))) / max(lam_O,1e-30)
