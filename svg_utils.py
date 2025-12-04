import numpy as np


def draw_schema_svg(Ls_ft, Ws_ft, ks_md, title="", canvas_width=980, canvas_height=220):
    """
    Build an SVG schematic for compartmentalized reservoir + continuous well.

    - Ls_ft: list/array of block lengths [ft]
    - Ws_ft: list/array of well lengths within each block [ft]
    - ks_md: list/array of permeabilities [md] (used for coloring)
    - title: text shown on top-left
    Returns: SVG string
    """
    Ls = np.asarray(Ls_ft, dtype=float)
    Ws = np.asarray(Ws_ft, dtype=float)
    ks = np.asarray(ks_md, dtype=float) if ks_md is not None else np.zeros_like(Ls)

    nB = len(Ls)
    if nB == 0:
        return (
            f'<svg width="{canvas_width}" height="{canvas_height}" style="background:#0b0b0b">'
            f'<text x="12" y="26" fill="#0f0" font-size="15">{title}</text>'
            f'<text x="12" y="56" fill="#bbb" font-size="13">Sin bloques para dibujar.</text>'
            f'</svg>'
        )

    margin_x, top_y, h_rect = 40, 80, 90
    total_L = float(np.sum(Ls))
    px_per_ft = (canvas_width - 2 * margin_x) / max(total_L, 1.0)

    def block_color(k):
        return "#96e07f" if float(k) >= 100.0 else "#cfcfcf"

    # Edges in ft and px
    x_ft_edges = np.concatenate(([0.0], np.cumsum(Ls)))
    x_px_edges = margin_x + x_ft_edges * px_per_ft

    # Draw blocks and labels
    rects, ticks, texts = [], [], []
    for i in range(nB):
        x0, x1 = x_px_edges[i], x_px_edges[i + 1]
        wpx = x1 - x0
        rects.append(
            f'<rect x="{x0:.1f}" y="{top_y}" width="{wpx:.1f}" height="{h_rect}" '
            f'style="fill:{block_color(ks[i])};stroke:#333;stroke-width:1"/>'
        )
        ticks.append(
            f'<line x1="{x0:.1f}" y1="{top_y}" x2="{x0:.1f}" y2="{top_y+h_rect}" style="stroke:#555;stroke-width:1"/>'
        )
        if i == nB - 1:
            ticks.append(
                f'<line x1="{x1:.1f}" y1="{top_y}" x2="{x1:.1f}" y2="{top_y+h_rect}" style="stroke:#555;stroke-width:1"/>'
            )
        cx = 0.5 * (x0 + x1)
        label = f"C{i+1}: L={Ls[i]:.0f} ft, k={ks[i]:.0f} md" if wpx >= 180 else (
            f"C{i+1}: k={ks[i]:.0f} md" if wpx >= 120 else f"C{i+1}"
        )
        texts.append(
            f'<text x="{cx:.1f}" y="{top_y-10}" fill="#8ff" font-size="13" text-anchor="middle">{label}</text>'
        )

    # Well base (whole chain)
    y_well = top_y + h_rect / 2
    well_base = (
        f'<line x1="{x_px_edges[0]:.1f}" y1="{y_well:.1f}" '
        f'x2="{x_px_edges[-1]:.1f}" y2="{y_well:.1f}" style="stroke:#2b2b2b;stroke-width:5"/>'
    )

    # Continuous chain: each block segment starts at clamped(prev_end)
    well_segments = []
    starts_px = []
    widths_px = []
    prev_end_px = x_px_edges[0]
    for i in range(nB):
        L_ft, W_ft = float(Ls[i]), max(0.0, float(Ws[i]))
        if W_ft <= 0:
            starts_px.append(None)
            widths_px.append(0.0)
            prev_end_px = max(prev_end_px, x_px_edges[i + 1])
            continue
        W_px = min(W_ft, L_ft) * px_per_ft
        start_px = max(x_px_edges[i], min(prev_end_px, x_px_edges[i + 1] - W_px))
        end_px = start_px + W_px
        well_segments.append(
            f'<rect x="{start_px:.1f}" y="{(y_well-3):.1f}" width="{W_px:.1f}" height="6" '
            f'style="fill:#1ea7ff;stroke:#0ff;stroke-width:0.7"/>'
        )
        starts_px.append(start_px)
        widths_px.append(W_px)
        prev_end_px = end_px

    # 40-ft ticks inside well segments within each block
    tick_elems, tick_labels = [], []
    for i in range(nB):
        L_ft, W_ft = float(Ls[i]), max(0.0, float(Ws[i]))
        if W_ft <= 0:
            continue
        x0_blk, x1_blk = x_px_edges[i], x_px_edges[i + 1]
        xs = starts_px[i]
        if xs is None:
            continue
        xe = xs + widths_px[i]
        # start of ticks: max(block start, well start in block)
        start_tick_px = max(x0_blk, xs)
        start_tick_ft = (start_tick_px - margin_x) / px_per_ft
        base_ft = max(x_ft_edges[i], start_tick_ft)
        tft = base_ft
        jcount = 0
        while True:
            if tft > (x_ft_edges[i + 1] - 1e-9) or (tft - start_tick_ft) > (W_ft + 1e-9):
                break
            tpx = margin_x + tft * px_per_ft
            tick_elems.append(
                f'<line x1="{tpx:.1f}" y1="{y_well-22:.1f}" x2="{tpx:.1f}" y2="{y_well+22:.1f}" '
                f'style="stroke:#ddd;stroke-width:1"/>'
            )
            jcount += 1
            tick_labels.append(
                f'<text x="{tpx:.1f}" y="{y_well+36:.1f}" fill="#ddd" font-size="12" text-anchor="middle">q{i+1}{jcount}</text>'
            )
            tft += 40.0

    title_txt = f'<text x="12" y="26" fill="#0f0" font-size="15">{title}</text>'
    footer = (
        '<text x="12" y="208" fill="#bbb" font-size="12">'
        'Bloques (gris/verde segun k). Pozo continuo (azul) encadenado sin saltos entre compartimentos.'
        '</text>'
    )

    svg = (
        f'<svg width="{canvas_width}" height="{canvas_height}" style="background:#0b0b0b">'
        f'{title_txt}'
        f'{"".join(rects)}{"".join(ticks)}{well_base}{"".join(well_segments)}{"        ".join(tick_elems)}{"".join(tick_labels)}{"".join(texts)}{footer}'
        f'</svg>'
    )
    return svg

