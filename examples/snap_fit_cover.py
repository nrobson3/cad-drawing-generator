"""Example: a drawing of a real moulded part from its STL export.

The part is the cover of a two-part snap-fit sensor housing (an original design,
see the fusion360-parametric-parts repo). Every view is a plane section of the
mesh: the plate outline and vents, the snap arms and PCB bosses on the
underside, section A-A through both arms, and a 5:1 detail of one arm.

    python examples/snap_fit_cover.py path/to/SH-200_Cover.stl

The STL is not in this repo: build the housing with snap_fit_housing.py in the
fusion360-parametric-parts repo and export the cover (SH-200) as STL, in mm.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from drawsheet import Sheet, View, dims, gdt, mesh  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def main(stl):
    tris = mesh.read_stl(stl)
    sheet = Sheet('A3', projection='first', title='SH-200  COVER', number='SH-200', rev='A',
                  scale='1.5:1', material='ABS (SABIC CYCOLAC MG47F)', finish='MT-11010 texture, A side',
                  drawn='N. Robson', date='2026-10-15', owner='original design')
    s = 1.5

    # top view: plate cut just under its front face (vents and service notches show)
    top = View(sheet, centre=(85, 205), scale=s)
    plate = mesh.section(tris, origin=(0, 0, 19.0), normal=(0, 0, 1), up=(0, 1, 0))
    top.outline(plate)
    top.centre_lines((0, 0), half=36)
    # A-A is cut on the plane normal +x and viewed looking along -x, so the arrows point -x
    top.cutting_plane((0, -36), (0, 36), 'A', look=(-1, 0))
    top.title('TOP   1.5:1', extent=31 * s, offset=32)
    # overall size, measured on this section (just under the front face)
    xs = [p[0] for outer, _ in plate for p in outer]
    ys = [p[1] for outer, _ in plate for p in outer if abs(p[0]) > 15]     # clear of the notches
    dims.linear(top, (min(xs), 0), (max(xs), 0), offset=-(31 * s + 18), direction='h')
    dims.linear(top, (-20, min(ys)), (-20, max(ys)), offset=-(31 * s - 20 * s + 10), direction='v')

    # underside: plate outline with the arms and bosses cut 1 mm below the plate
    under = View(sheet, centre=(225, 205), scale=s, flip=(1, -1))
    under.outline(plate)
    feats = mesh.section(tris, origin=(0, 0, 17.0), normal=(0, 0, 1), up=(0, 1, 0))
    under.section(feats, spacing=1.6)
    under.centre_lines((0, 0), half=36)
    under.title('UNDERSIDE   1.5:1', extent=31 * s, offset=12)

    # section A-A through both snap arms (plane x = 0), y across, z up
    sec = View(sheet, centre=(85, 96), scale=s)
    cut = mesh.section(tris, origin=(0, 0, 12.0), normal=(1, 0, 0), up=(0, 0, 1))
    sec.section(cut, spacing=1.6)
    sec.title('SECTION A-A   1.5:1', extent=10 * s, offset=12)
    dc, r = (365.0, 112.0), 33.0
    sec.detail_marker((28.3, -0.5), r / 5.0, 'C')          # the region detail C shows at 5:1
    # plate thickness, measured on the centre piece of the section (between the vents)
    centre_piece = next(o for o, _ in cut if min(p[0] for p in o) < 0 < max(p[0] for p in o)
                        and min(p[1] for p in o) > 0)
    v_lo, v_hi = min(p[1] for p in centre_piece), max(p[1] for p in centre_piece)
    dims.leader(sec, (8, v_hi), (128, 118), '%s THK' % dims.fmt(v_hi - v_lo))
    gdt.datum_feature(sheet, sec.to_paper((-14, v_lo)), 'A', direction=(0, -1), length=8)

    # detail C, 5:1, clipped to a circle round one arm
    det = View(sheet, centre=(dc[0] - 28.3 * 5, dc[1] + 0.5 * 5), scale=5.0, clip=(dc[0], dc[1], r))
    det.section(cut, spacing=2.0)
    sheet.circle(dc, r, width=0.25)
    sheet.text(dc[0], dc[1] - r - 7, 'DETAIL C   5:1', cap=3.5, ha='center', weight='bold')
    x_f, y_f = 200, 118
    gdt.feature_control_frame(sheet, x_f, y_f, 'profile_surface', '0.05', datums=('A',))
    sheet.text(x_f, y_f - 5, 'CATCH FACE TO DATUM A', cap=2.5)

    sheet.notes([
        'MATERIAL ABS. STEEL SIZED FOR 0.6 % SHRINK.',
        'DRAFT PER 3D MODEL: WALLS 1.5 DEG, VENTS 1.0, BOSSES 0.5.',
        'ALL VIEWS ARE PLANE SECTIONS OF THE STL; MODEL IS MASTER.',
    ], y=60)
    os.makedirs(OUT, exist_ok=True)
    out = sheet.save(os.path.join(OUT, 'snap_fit_cover.png'), dpi=110)
    sheet.save(os.path.join(OUT, 'snap_fit_cover.pdf'))
    sheet.close()
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: python examples/snap_fit_cover.py path/to/cover.stl')
    print(main(sys.argv[1]))
