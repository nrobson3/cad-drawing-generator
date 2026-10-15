"""Tests for drawsheet. Run from the repository root: python -m unittest discover tests"""
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'examples'))

from drawsheet import Sheet, View, dims, gdt, mesh  # noqa: E402
from drawsheet.sheet import SIZES, font_pt, text_width, zone_count  # noqa: E402
from drawsheet.view import circle_polygon, clip_convex, compound_path, signed_area  # noqa: E402
import flanged_bushing  # noqa: E402


def box(x0, y0, z0, x1, y1, z1):
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([(v[a], v[b], v[c]) for a, b, c, d in q] + [(v[a], v[c], v[d]) for a, b, c, d in q], float)


class SheetTests(unittest.TestCase):
    def test_sizes_and_zones(self):
        self.assertEqual(SIZES['A3'], (420.0, 297.0))
        for length in (100, 277, 390, 811):
            n = zone_count(length)
            self.assertEqual(n % 2, 0)
            self.assertLess(abs(length / n - 50), 25)

    def test_font_size_from_cap_height(self):
        self.assertAlmostEqual(font_pt(3.5) * 0.72 * 25.4 / 72, 3.5)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            Sheet('B5')
        with self.assertRaises(ValueError):
            Sheet('A4', projection='second')

    def test_saves_vector_and_raster(self):
        sh = Sheet('A4', title='TEST', number='T-1')
        d = tempfile.mkdtemp()
        for ext in ('pdf', 'svg', 'png'):
            p = sh.save(os.path.join(d, 'x.' + ext), dpi=60)
            self.assertGreater(os.path.getsize(p), 1000)
        sh.close()


class ViewTests(unittest.TestCase):
    def test_mapping_scale_axes_flip(self):
        sh = Sheet('A4')
        v = View(sh, centre=(100, 50), scale=2.0, axes=('y', 'z'), flip=(1, -1))
        self.assertEqual(v.to_paper((9, 3, 4)), (106.0, 42.0))
        self.assertEqual(v.to_paper((3, 4)), (106.0, 42.0))      # 2D points use (x, y) directly
        self.assertAlmostEqual(v.model_length(10), 5.0)
        sh.close()

    def test_compound_path_orients_holes(self):
        outer = [(0, 0), (0, 10), (10, 10), (10, 0)]           # clockwise on purpose
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]                # counter-clockwise on purpose
        path = compound_path([(outer, [hole])])
        # the nonzero fill rule leaves the hole empty only if the rings wind opposite ways
        self.assertGreater(signed_area(path.vertices[:4]), 0)
        self.assertLess(signed_area(path.vertices[5:9]), 0)
        self.assertEqual(list(path.codes).count(path.CLOSEPOLY), 2)

    def test_clip_convex_against_circle(self):
        window = circle_polygon((0, 0), 5, n=256)
        sq = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        self.assertAlmostEqual(abs(signed_area(clip_convex(sq, window))), math.pi * 25, delta=0.05)
        inside = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        self.assertAlmostEqual(abs(signed_area(clip_convex(inside, window))), 4.0)
        self.assertEqual(clip_convex([(20, 20), (21, 20), (21, 21)], window), [])


class DimTests(unittest.TestCase):
    def test_formatting(self):
        self.assertEqual(dims.fmt(12.0), '12')
        self.assertEqual(dims.fmt(12.50), '12.5')
        self.assertEqual(dims.fmt(-0.0001), '0')
        self.assertEqual(dims.tolerance_text(10, 0.05), '10 ±0.05')
        self.assertEqual(dims.tolerance_text(12, (0.018, 0)), '12 +0.018/0')
        self.assertEqual(dims.tolerance_text(24, (-0.007, -0.020)), '24 -0.007/-0.020')    # g6 shaft
        self.assertEqual(dims.tolerance_text(30, (0.1, -0.05)), '30 +0.10/-0.05')       # equal decimals
        self.assertEqual(dims.fmt(40, 0), '40')
        self.assertEqual(dims.fmt(100.0, 1), '100')
        self.assertEqual(dims.tolerance_text(30, (0.1, -0.1)), '30 +0.1/-0.1')

    def test_measures_model_length_at_any_scale(self):
        sh = Sheet('A4')
        for scale in (0.5, 1.0, 2.0, 5.0):
            v = View(sh, centre=(100, 100), scale=scale)
            self.assertAlmostEqual(dims.linear(v, (0, 0), (37.25, 4), offset=10, direction='h'), 37.25)
            self.assertAlmostEqual(dims.linear(v, (0, 0), (3, 4), offset=10, direction='a'), 5.0)
        sh.close()

    def test_vertical_offset_sign(self):
        sh = Sheet('A4')
        v = View(sh, centre=(100, 100), scale=1.0)
        n = len(sh.ax.texts)
        dims.linear(v, (0, 0), (0, 20), offset=-10, direction='v')
        self.assertLess(sh.ax.texts[n].get_position()[0], 100)   # negative is to the left
        sh.close()


class GdtTests(unittest.TestCase):
    def test_every_symbol_draws(self):
        sh = Sheet('A4')
        for i, kind in enumerate(sorted(gdt.CHARACTERISTICS)):
            datums = ('A',) if kind in gdt.NEEDS_DATUM else ()
            right = gdt.feature_control_frame(sh, 30, 20 + i * 9, kind, '0.05', datums=datums)
            self.assertGreater(right, 30 + 14)
        self.assertEqual(len(gdt.CHARACTERISTICS), 14)
        sh.close()

    def test_orientation_needs_datum(self):
        sh = Sheet('A4')
        with self.assertRaises(ValueError):
            gdt.feature_control_frame(sh, 10, 10, 'perpendicularity', '0.1')
        with self.assertRaises(ValueError):
            gdt.feature_control_frame(sh, 10, 10, 'roundness', '0.1')
        sh.close()


class MeshTests(unittest.TestCase):
    def test_box_section(self):
        t = box(0, 0, 0, 20, 10, 5)
        s = mesh.section(t, (0, 0, 2.5), (0, 0, 1))
        self.assertEqual(len(s), 1)
        self.assertAlmostEqual(mesh.area(s), 200.0, places=6)

    def test_hollow_section_nests_hole(self):
        tris = flanged_bushing.revolve([(6, 0), (12, 0), (12, 20), (6, 20)], segments=256)
        s = mesh.section(tris, (0, 0, 10), (0, 0, 1))
        self.assertEqual(len(s), 1)
        self.assertEqual(len(s[0][1]), 1)
        n = 256                                     # regular n-gons: area = n r^2 sin(2 pi / n) / 2
        ring = n / 2 * math.sin(2 * math.pi / n) * (144 - 36)
        self.assertAlmostEqual(mesh.area(s), ring, places=6)

    def test_axial_section_matches_profile(self):
        tris = flanged_bushing.revolve(flanged_bushing.PROFILE)
        s = mesh.section(tris, (0, 0, 0), (0, 1, 0), up=(1, 0, 0))
        p = flanged_bushing.PROFILE
        half = 0.5 * abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                             for i in range(len(p))))
        self.assertAlmostEqual(mesh.area(s), 2 * half, places=6)
        self.assertEqual(len(s), 2)                                   # two halves either side of the bore

    def test_plane_basis_is_right_handed(self):
        for n in ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
            u, v, w = mesh.plane_basis(n)
            self.assertAlmostEqual(float(np.dot(np.cross(u, v), w)), 1.0)

    def test_stl_roundtrip(self):
        t = box(0, 0, 0, 3, 4, 5)
        d = tempfile.mkdtemp()
        b = os.path.join(d, 'b.stl')
        with open(b, 'wb') as f:
            f.write(b'\0' * 80 + struct.pack('<I', len(t)))
            for tri in t:
                f.write(struct.pack('<12fH', 0, 0, 0, *tri.ravel(), 0))
        a = os.path.join(d, 'a.stl')
        with open(a, 'w') as f:
            f.write('solid x\n')
            for tri in t:
                f.write('facet normal 0 0 0\nouter loop\n' + ''.join('vertex %f %f %f\n' % tuple(p) for p in tri)
                        + 'endloop\nendfacet\n')
            f.write('endsolid x\n')
        for path in (a, b):
            r = mesh.read_stl(path)
            self.assertEqual(r.shape, (12, 3, 3))
            self.assertAlmostEqual(mesh.area(mesh.section(r, (0, 0, 2), (0, 0, 1))), 12.0, places=5)


    def test_island_inside_a_hole(self):
        # tube in a tube: outer ring with a hole, and a separate ring inside that hole
        rev = flanged_bushing.revolve
        t = np.vstack([rev([(8, 0), (10, 0), (10, 5), (8, 5)], 64), rev([(2, 0), (4, 0), (4, 5), (2, 5)], 64)])
        s = mesh.section(t, (0, 0, 2.5), (0, 0, 1))
        self.assertEqual(len(s), 2)
        self.assertEqual(sorted(len(h) for _, h in s), [1, 1])

    def test_open_mesh_warns(self):
        t = box(0, 0, 0, 10, 10, 10)
        t = np.array([tri for tri in t if not np.allclose(tri[:, 0], 0)])     # drop the x = 0 face
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            s = mesh.section(t, (5, 5, 5), (0, 0, 1))
        self.assertTrue(any('open chain' in str(x.message) for x in w))
        self.assertEqual(s, [])

    def test_plane_through_mesh_vertices_closes(self):
        # the cut at angle 0 runs through a whole column of vertices of the revolved
        # mesh; the sliver segments there must not split or open the loops
        t = flanged_bushing.revolve(flanged_bushing.PROFILE, 36)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            s = mesh.section(t, (0, 0, 0), (0, 1, 0), up=(1, 0, 0))
        self.assertEqual(len(s), 2)
        self.assertTrue(all(h == [] for _, h in s))

    def test_section_through_a_vertex_plane(self):
        t = box(0, 0, 0, 20, 10, 5)
        with warnings.catch_warnings():              # a cut exactly on a face is documented as unreliable
            warnings.simplefilter('ignore')
            s = mesh.section(t, (0, 0, 0), (1, 0, 0), up=(0, 0, 1))
        self.assertLessEqual(mesh.area(s), 50.0 + 1e-9)          # never more than the face itself

    def test_bad_stl_raises(self):
        p = os.path.join(tempfile.mkdtemp(), 'bad.stl')
        with open(p, 'w') as f:
            f.write('solid nothing\nendsolid\n')
        with self.assertRaises(ValueError):
            mesh.read_stl(p)

    def test_plane_touching_a_ridge_has_no_area(self):
        # a ring of circular section, cut where it is only just touched: the two
        # cut loops are a fraction of a micron apart and must not collapse into one
        circle = [(10 + 3 * math.sin(math.radians(a)), 3 * math.cos(math.radians(a))) for a in range(0, 360, 15)]
        t = flanged_bushing.revolve(circle[::-1], 90)
        k = math.sin(math.radians(15)) / (1 - math.cos(math.radians(15)))   # dr/dz of the edges at the ridge
        ring = 90 / (2 * math.pi) * math.sin(2 * math.pi / 90)               # 90-gon against the circle
        for dz in (0.0, 1e-7, 2e-6, 1e-4):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a = mesh.area(mesh.section(t, (0, 0, 3.0 - dz), (0, 0, 1)))
            self.assertAlmostEqual(a, 2 * math.pi * 10 * 2 * k * dz * ring, delta=1e-6 + 0.01 * a)   # 1e-6: the on-plane nudge

    def test_bodies_touching_at_a_corner_stay_separate(self):
        t = np.vstack([box(0, 0, 0, 1, 1, 1), box(1, 1, 0, 2, 2, 1)])
        rng = np.random.default_rng(1)
        for _ in range(10):
            s = mesh.section(t[rng.permutation(len(t))], (0, 0, 0.5), (0, 0, 1))
            self.assertEqual(len(s), 2)
            self.assertAlmostEqual(mesh.area(s), 2.0, places=9)

    def test_bodies_sharing_a_face(self):
        t = np.vstack([box(0, 0, 0, 1, 1, 1), box(1, 0, 0, 2, 1, 1)])
        rng = np.random.default_rng(2)
        for _ in range(10):
            s = mesh.section(t[rng.permutation(len(t))], (0, 0, 0.5), (0, 0, 1))
            self.assertAlmostEqual(mesh.area(s), 2.0, places=9)

    def test_binary_stl_with_a_solid_header(self):
        tris = box(0, 0, 0, 1, 2, 3)
        p = os.path.join(tempfile.mkdtemp(), 'b.stl')
        with open(p, 'wb') as f:
            f.write(b'solid exported by some CAD'.ljust(80, b' ') + struct.pack('<I', len(tris)))
            for tri in tris:
                f.write(struct.pack('<12fH', 0, 0, 0, *tri.ravel(), 0))
            f.write(b'\0' * 16)                      # trailing padding
        self.assertEqual(mesh.read_stl(p).shape, tris.shape)

    def test_bad_planes_are_refused(self):
        with self.assertRaises(ValueError):
            mesh.section(box(0, 0, 0, 1, 1, 1), (0, 0, 0.5), (0, 0, 0))
        with self.assertRaises(ValueError):
            mesh.plane_basis((0, 0, 1), up=(0, 0, 2))

    def test_a_plane_that_misses_draws_nothing(self):
        s = mesh.section(box(0, 0, 0, 1, 1, 1), (0, 0, 9), (0, 0, 1))
        self.assertEqual(s, [])
        v = View(Sheet('A4'), centre=(100, 100))
        self.assertIsNone(v.section(s))
        self.assertIsNone(v.outline(s))

    def test_numpy_numbers_as_tolerances(self):
        self.assertEqual(dims.tolerance_text(10, np.float32(0.05)), '10 ±0.05')
        self.assertEqual(dims.tolerance_text(10, np.int64(1)), '10 ±1')
        self.assertEqual(dims.tolerance_text(12, (0.0004, 0)), '12 0/0')
        with self.assertRaises(ValueError):
            dims.tolerance_text(10, -0.05)


class LayoutTests(unittest.TestCase):
    def test_text_width_is_measured(self):
        self.assertGreater(text_width('0.05', 3.5), text_width('0.5', 3.5))
        self.assertAlmostEqual(text_width('', 3.5), 0.0)
        self.assertLess(text_width('0.05', 3.5), 4 * 3.5)

    def test_frame_margins(self):
        a3, a1 = Sheet('A3'), Sheet('A1')
        self.assertEqual((a3.left, a3.bottom, a3.w - a3.right), (20.0, 10.0, 10.0))
        self.assertEqual((a1.left, a1.bottom, a1.w - a1.right), (20.0, 20.0, 20.0))

    def test_projection_symbols(self):
        for method in ('first', 'third'):
            sh = Sheet('A4', projection=method)
            polys = [p for p in sh.ax.patches if p.__class__.__name__ == 'Polygon']
            circles = [p for p in sh.ax.patches if p.__class__.__name__ == 'Circle']
            cone_x = np.mean(polys[-1].get_xy()[:, 0])
            circ_x = circles[-1].center[0]
            self.assertEqual(cone_x < circ_x, method == 'first')

    def test_detail_view_hatch_stays_in_the_circle(self):
        sh = Sheet('A4')
        sq = [([(-10, -10), (10, -10), (10, 10), (-10, 10)], [])]
        v = View(sh, centre=(100, 100), scale=1.0, clip=(100, 100, 5))
        v.section(sq)
        hatch = sh.ax.collections[-1]
        clip = hatch.get_clip_path().get_fully_transformed_path()
        corners = sh.ax.transData.inverted().transform(clip.vertices)
        # the hatch is clipped to the square cut down to the r = 5 circle
        self.assertLessEqual(np.hypot(corners[:, 0] - 100, corners[:, 1] - 100).max(), 5.0 + 1e-6)


class ExampleTests(unittest.TestCase):
    def test_bushing_example_runs(self):
        pdf, png, area = flanged_bushing.main()
        self.assertTrue(os.path.exists(pdf) and os.path.exists(png))
        self.assertAlmostEqual(area, 526.0, places=6)


if __name__ == '__main__':
    unittest.main()
