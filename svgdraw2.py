import svgwrite

# ======== FIGURE 1 — Trilinear Model ===========
def draw_fig1_trilinear():
    dwg = svgwrite.Drawing("fig1_trilinear.svg", size=("900px", "400px"))

    # Outer Reservoir
    dwg.add(dwg.rect(insert=(400, 100), size=(300, 200),
                     fill="#a3c9f9", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Outer Reservoir", insert=(500, 90), font_size="14px"))
    dwg.add(dwg.text("kₒ, φₒ, cₜₒ", insert=(500, 320), font_size="12px"))

    # Inner Reservoir
    dwg.add(dwg.rect(insert=(250, 120), size=(150, 160),
                     fill="#d9d9d9", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Inner Reservoir", insert=(265, 100), font_size="14px"))
    dwg.add(dwg.text("kᵢ, φᵢ, cₜᵢ", insert=(280, 310), font_size="12px"))

    # Hydraulic Fracture
    dwg.add(dwg.rect(insert=(230, 160), size=(20, 80),
                     fill="#fcb75d", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Hydraulic Fracture", insert=(100, 100), font_size="14px"))
    dwg.add(dwg.text("k_F, φ_F, cₜ_F", insert=(100, 310), font_size="12px"))

    # Horizontal Well
    dwg.add(dwg.line(start=(230, 200), end=(150, 200),
                     stroke="black", stroke_width=3))
    dwg.add(dwg.text("Horizontal Well", insert=(50, 220), font_size="12px"))

    # Flow arrows (→)
    for x in range(250, 700, 50):
        dwg.add(dwg.line(start=(x, 200), end=(x + 30, 200),
                         stroke="black", stroke_width=1.5))
        dwg.add(dwg.polygon([(x+30, 200), (x+25, 195), (x+25, 205)],
                            fill="black"))

    # Discretization grid
    for x in range(250, 700, 25):
        dwg.add(dwg.line(start=(x, 120), end=(x, 280),
                         stroke="#cccccc", stroke_width=0.5))
    for y in range(120, 281, 20):
        dwg.add(dwg.line(start=(250, y), end=(700, y),
                         stroke="#cccccc", stroke_width=0.5))

    dwg.save()
    print("✅ fig1_trilinear.svg generated")

# ======== FIGURE 2 — Symmetry & Multiple Fractures ===========
def draw_fig2_symmetry():
    dwg = svgwrite.Drawing("fig2_symmetry.svg", size=("800px", "400px"))

    # Box for full domain
    dwg.add(dwg.rect(insert=(100, 80), size=(600, 240),
                     fill="#e6f2ff", stroke="black", stroke_width=1))
    dwg.add(dwg.text("h", insert=(720, 200), font_size="14px"))

    # Hydraulic fractures (gray plates)
    for i in range(0, 5):
        x = 150 + i * 120
        dwg.add(dwg.rect(insert=(x, 80), size=(10, 240),
                         fill="#999999", stroke="black", stroke_width=1))

    dwg.add(dwg.text("d_F", insert=(160, 70), font_size="12px"))
    dwg.add(dwg.text("w_F", insert=(260, 70), font_size="12px"))
    dwg.add(dwg.text("x_F", insert=(400, 70), font_size="12px"))
    dwg.add(dwg.text("x_e", insert=(580, 70), font_size="12px"))

    # Symmetry element
    dwg.add(dwg.rect(insert=(720, 100), size=(60, 200),
                     fill="#b0c4de", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Symmetry Element", insert=(700, 330), font_size="13px"))

    dwg.save()
    print("✅ fig2_symmetry.svg generated")

# ======== FIGURE 3 — 3D SRV / ORV Representation ===========
def draw_fig3_reservoirs():
    dwg = svgwrite.Drawing("fig3_reservoirs.svg", size=("950px", "500px"))

    # Outer Reservoir
    dwg.add(dwg.rect(insert=(550, 200), size=(300, 150),
                     fill="#b5d0ff", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Outer Reservoir", insert=(630, 190), font_size="14px"))

    # Inner Reservoir
    dwg.add(dwg.rect(insert=(400, 220), size=(150, 130),
                     fill="#cccccc", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Inner Reservoir", insert=(420, 210), font_size="14px"))

    # Fractures
    for i in range(0, 4):
        x = 380 - i * 30
        dwg.add(dwg.rect(insert=(x, 230), size=(20, 110),
                         fill="#ffb266", stroke="black", stroke_width=1))

    dwg.add(dwg.text("Finite-Conductivity Fractures", insert=(80, 210), font_size="13px"))

    # Horizontal Well
    dwg.add(dwg.line(start=(350, 285), end=(150, 285),
                     stroke="black", stroke_width=3))
    dwg.add(dwg.text("Horizontal Well", insert=(80, 300), font_size="13px"))

    # SRV and ORV small blocks
    for i in range(0, 3):
        dwg.add(dwg.rect(insert=(600 + i * 50, 370), size=(40, 30),
                         fill="#e0ebf5", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Matrix slabs", insert=(600, 420), font_size="12px"))

    dwg.save()
    print("✅ fig3_reservoirs.svg generated")

# ======== RUN ALL ===========
draw_fig1_trilinear()
draw_fig2_symmetry()
draw_fig3_reservoirs()
print("\nAll SVG figures generated successfully.")
