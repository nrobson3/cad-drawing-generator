"""Example: an A3 drawing of a flanged bushing, generated end to end in code.

The part is built as a mesh by revolving its half-profile, cut into sections with
drawsheet.mesh, and drawn: full section A-A, end view, dimensions, a datum on the
bore and three geometric tolerances.

    python examples/flanged_bushing.py   ->  examples/out/flanged_bushing.pdf / .png
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from drawsheet import Sheet, View, dims, gdt, mesh  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')

# half-profile in (r, z), counter-clockwise; z = 0 is the flange face
FLANGE_D, FLANGE_T = 40.0, 6.0
BODY_D, LENGTH = 24.0, 36.0
BORE_D = 12.0
CHAMFER = 1.0
PROFILE = [
    (BORE_D / 2 + CHAMFER, 0.0), (FLANGE_D / 2, 0.0), (FLANGE_D / 2, FLANGE_T),
    (BODY_D / 2, FLANGE_T), (BODY_D / 2, LENGTH - CHAMFER), (BODY_D / 2 - CHAMFER, LENGTH),
    (BORE_D / 2, LENGTH), (BORE_D / 2, CHAMFER),
]


def revolve(profile, segments=180):
    """Triangles of the solid swept by a closed (r, z) profile about the z axis."""
    a = np.linspace(0, 2 * math.pi, segments + 1)
    tris = []
    for i in range(len(profile)):
        (r0, z0), (r1, z1) = profile[i], profile[(i + 1) % len(profile)]
        for k in range(segments):
            c0, s0, c1, s1 = math.cos(a[k]), math.sin(a[k]), math.cos(a[k + 1]), math.sin(a[k + 1])
            p00, p01 = (r0 * c0, r0 * s0, z0), (r0 * c1, r0 * s1, z0)
            p10, p11 = (r1 * c0, r1 * s0, z1), (r1 * c1, r1 * s1, z1)
            tris += [(p00, p11, p10), (p00, p01, p11)]         # outward, as an STL export
    return np.array(tris, float)


def main():
    tris = revolve(PROFILE)
    sheet = Sheet('A3', projection='first', title='FLANGED BUSHING', number='EX-001', rev='A',
                  scale='2:1', material='AISI 303 STAINLESS', finish='Ra 1.6 bore, Ra 3.2 elsewhere',
                  drawn='N. Robson', date='2026-10-15', owner='drawsheet example')
    s = 2.0

    # SECTION A-A: the cut through the axis, axis horizontal on paper (z -> paper x)
    cut = mesh.section(tris, origin=(0, 0, 0), normal=(0, 1, 0), up=(1, 0, 0))   # u = z, v = x
    sec = View(sheet, centre=(70, 170), scale=s)
    sec.section(cut, spacing=2.5)
    sec.lines([[(-4, 0), (LENGTH + 4, 0)]], width=0.18, style=(0, (12, 3, 2, 3)))     # axis
    sec.title('SECTION A-A   2:1', extent=FLANGE_D / 2 * s, offset=18)

    # END VIEW. First angle puts the view from the LEFT (the flange end) on the right
    # of A-A. From there the flange face, its bore chamfer and the bore are visible;
    # the 24 body is behind the flange, so it is hidden (dashed).
    end = View(sheet, centre=(250, 170), scale=s)
    face = mesh.section(tris, origin=(0, 0, 0.001), normal=(0, 0, 1), up=(0, 1, 0))      # 40 / 14 chamfer edge
    bore = mesh.section(tris, origin=(0, 0, FLANGE_T / 2), normal=(0, 0, 1), up=(0, 1, 0))
    body = mesh.section(tris, origin=(0, 0, (FLANGE_T + LENGTH) / 2), normal=(0, 0, 1), up=(0, 1, 0))
    end.outline(face)
    end.lines([h + [h[0]] for _, holes in bore for h in holes], width=0.5)               # bore edge
    end.hidden([o + [o[0]] for o, _ in body])                                            # body, behind
    end.centre_mark((0, 0), FLANGE_D / 2)
    # A-A is cut on the plane normal +y and viewed looking along -y, so the arrows point -y
    end.cutting_plane((-FLANGE_D / 2 - 6, 0), (FLANGE_D / 2 + 6, 0), 'A', look=(0, -1))
    end.title('END VIEW   2:1', extent=FLANGE_D / 2 * s, offset=14)

    # dimensions on the section (model coordinates: (z, r))
    dims.linear(sec, (0, FLANGE_D / 2), (LENGTH, BODY_D / 2), offset=14, direction='h')
    dims.linear(sec, (0, -FLANGE_D / 2), (FLANGE_T, -FLANGE_D / 2), offset=-10, direction='h')
    dims.linear(sec, (0, -FLANGE_D / 2), (0, FLANGE_D / 2), offset=-12, direction='v',
                text=chr(0x2300) + dims.fmt(FLANGE_D))
    dims.linear(sec, (LENGTH, -BODY_D / 2), (LENGTH, BODY_D / 2), offset=12, direction='v',
                text=chr(0x2300) + dims.fmt(BODY_D) + ' g6')
    dims.linear(sec, (LENGTH * 0.6, -BORE_D / 2), (LENGTH * 0.6, BORE_D / 2), offset=0, direction='v',
                text=chr(0x2300) + '12 H7')
    dims.leader(sec, (LENGTH - CHAMFER / 2, BODY_D / 2 - CHAMFER / 2), (175, 205), '2x 1 x 45' + chr(0xB0))

    # datum A on the bore, and the tolerances that hang off it
    # on the size dimension of the bore, so the datum is the bore's axis
    gdt.datum_feature(sheet, sec.to_paper((LENGTH * 0.6, -BORE_D / 2)), 'A', direction=(0, -1), length=22)
    x_fcf, y_fcf = 40, 92
    gdt.feature_control_frame(sheet, x_fcf, y_fcf, 'circular_runout', '0.02', datums=('A',))
    sheet.text(x_fcf, y_fcf - 5, chr(0x2300) + '24 g6 TO DATUM A', cap=2.5)
    gdt.feature_control_frame(sheet, x_fcf + 70, y_fcf, 'perpendicularity', '0.03', datums=('A',))
    sheet.text(x_fcf + 70, y_fcf - 5, 'FLANGE FACE TO DATUM A', cap=2.5)
    gdt.feature_control_frame(sheet, x_fcf + 140, y_fcf, 'flatness', '0.01')
    sheet.text(x_fcf + 140, y_fcf - 5, 'FLANGE FACE', cap=2.5)

    sheet.notes([
        'BREAK SHARP EDGES 0.2 MAX UNLESS SHOWN.',
        'BORE 12 H7 IS DATUM A; TOLERANCES ARE ISO 1101.',
        'GENERAL TOLERANCES ISO 2768-mK.',
        'PART GENERATED IN CODE; THE DRAWING IS CUT FROM ITS MESH.',
    ], y=72)

    os.makedirs(OUT, exist_ok=True)
    pdf = sheet.save(os.path.join(OUT, 'flanged_bushing.pdf'))
    png = sheet.save(os.path.join(OUT, 'flanged_bushing.png'), dpi=110)
    sheet.close()
    return pdf, png, mesh.area(cut)


if __name__ == '__main__':
    pdf, png, a = main()
    print(pdf)
    print(png)
    print('section area %.2f mm2' % a)
