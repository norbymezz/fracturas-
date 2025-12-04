# Aplicación base para rehacer los ejemplos SVG sin valores hardcodeados
# (basado en los errores anteriores de main_102834.py)

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Reservorio SVG Interactivo", layout="wide")
st.title("🧭 Esquemas SVG — Modelos de Reservorio Interactivo")

# Entrada flexible para dimensiones generales
st.sidebar.header("Parámetros del Reservorio")
L_total = st.sidebar.number_input("Longitud total (ft)", 100.0, 2000.0, 450.0, 10.0)
H_total = st.sidebar.number_input("Altura (ft)", 10.0, 200.0, 30.0, 5.0)
W_total = st.sidebar.number_input("Ancho (ft)", 10.0, 200.0, 30.0, 5.0)

# Tabla editable para los bloques (sin hardcodear)
def default_blocks():
    return pd.DataFrame([
        {"Block": 1, "Block_length_ft": 250, "Well_length_ft": 150, "Well_segments": 3},
        {"Block": 2, "Block_length_ft": 200, "Well_length_ft": 50, "Well_segments": 1},
    ])

st.sidebar.write("---")

st.subheader("📋 Configuración de Bloques")
blocks = st.data_editor(default_blocks(), num_rows="dynamic")

# Validaciones
if blocks["Block_length_ft"].sum() > L_total:
    st.error("⚠️ La suma de las longitudes de los bloques excede la longitud total del reservorio.")

# Dibujo SVG (sin hardcodear)
st.subheader("🧩 Vista esquemática SVG")
W, H = 900, 240
margin = 40
scale = (W - 2 * margin) / L_total

# Generar SVG dinámico
def draw_svg(blocks, L_total):
    x0 = margin
    svg = [f'<svg width="{W}" height="{H}" style="background:#111">']

    # Dibujar bloques
    for i, row in blocks.iterrows():
        L = row["Block_length_ft"] * scale
        color = "#66bb6a" if i % 2 == 0 else "#42a5f5"
        svg.append(f'<rect x="{x0}" y="80" width="{L}" height="60" fill="{color}" stroke="#fff" stroke-width="1"/>')
        label = f'B{int(row.Block)}: {int(row.Block_length_ft)} ft'
        svg.append(f'<text x="{x0 + L/2}" y="75" fill="#fff" font-size="14" text-anchor="middle">{label}</text>')

        # Pozo dentro del bloque
        Ww = min(row["Well_length_ft"], row["Block_length_ft"]) * scale
        wx = x0 + (row["Block_length_ft"] - row["Well_length_ft"]) * 0.5 * scale
        svg.append(f'<rect x="{wx}" y="108" width="{Ww}" height="4" fill="#00bcd4"/>')

        # Segmentos del pozo
        seg_len = row["Well_length_ft"] / max(1, row["Well_segments"])
        for j in range(row["Well_segments"]):
            sx = wx + j * seg_len * scale
            svg.append(f'<line x1="{sx}" y1="104" x2="{sx}" y2="116" stroke="#ff9800" stroke-width="2"/>')

        x0 += L

    # Ejes
    svg.append(f'<line x1="{margin}" y1="160" x2="{W-margin}" y2="160" stroke="#fff" stroke-width="2"/>')
    svg.append(f'<text x="{W/2}" y="180" fill="#ccc" font-size="12">x (ft)</text>')

    svg.append('</svg>')
    return "\n".join(svg)

svg_code = draw_svg(blocks, L_total)
st.components.v1.html(svg_code, height=260, scrolling=False)

# Mostrar tabla final
st.write("---")
st.write("🧮 Datos finales:")
st.dataframe(blocks)

# Vista unificada (usa svg_utils.draw_schema_svg)
st.write("---")
st.subheader("Esquema SVG (unificado)")
try:
    from svg_utils import draw_schema_svg
    Ls2 = blocks["Block_length_ft"].astype(float).values
    Ws2 = blocks["Well_length_ft"].astype(float).values
    if "k_md" in blocks.columns:
        ks2 = blocks["k_md"].astype(float).values
    else:
        ks2 = np.where((blocks.index.values % 2) == 0, 10.0, 150.0)
    svg2 = draw_schema_svg(Ls2, Ws2, ks2, title="Reservorio SVG (unificado)")
    st.components.v1.html(svg2, height=260, scrolling=False)
except Exception as e:
    st.info(f"Vista unificada no disponible: {e}")
