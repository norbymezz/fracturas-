import svgwrite

def draw_hydraulic_system():
    dwg = svgwrite.Drawing("hydraulic_domains.svg", size=("900px", "400px"))
    
    # Outer Reservoir (light blue)
    dwg.add(dwg.rect(insert=(400, 100), size=(300, 200),
                     fill="#a3c9f9", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Outer Reservoir", insert=(500, 90), font_size="14px"))
    dwg.add(dwg.text("kₒ, φₒ, cₜₒ", insert=(500, 320), font_size="12px"))

    # Inner Reservoir (gray)
    dwg.add(dwg.rect(insert=(250, 120), size=(150, 160),
                     fill="#d9d9d9", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Inner Reservoir", insert=(265, 100), font_size="14px"))
    dwg.add(dwg.text("kᵢ, φᵢ, cₜᵢ", insert=(280, 310), font_size="12px"))

    # Hydraulic Fracture (orange)
    dwg.add(dwg.rect(insert=(230, 160), size=(20, 80),
                     fill="#fcb75d", stroke="black", stroke_width=1))
    dwg.add(dwg.text("Hydraulic Fracture", insert=(100, 100), font_size="14px"))
    dwg.add(dwg.text("k_F, φ_F, cₜ_F", insert=(100, 310), font_size="12px"))

    # Horizontal well
    dwg.add(dwg.line(start=(230, 200), end=(150, 200), stroke="black", stroke_width=3))
    dwg.add(dwg.text("Horizontal Well", insert=(50, 220), font_size="12px"))

    # Flow arrows
    for x in range(250, 700, 60):
        dwg.add(dwg.line(start=(x, 200), end=(x+30, 200),
                         stroke="black", stroke_width=1.5, marker_end=dwg.marker(id="arrow")))

    # Discretization grid
    for x in range(250, 700, 25):
        dwg.add(dwg.line(start=(x, 120), end=(x, 280), stroke="#bbbbbb", stroke_width=0.5))

    for y in range(120, 281, 20):
        dwg.add(dwg.line(start=(250, y), end=(700, y), stroke="#bbbbbb", stroke_width=0.5))

    # Save
    dwg.save()
    print("✅ SVG file generated: hydraulic_domains.svg")

draw_hydraulic_system()
