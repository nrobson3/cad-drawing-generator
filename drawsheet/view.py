"""Views: place model geometry on the sheet at a scale.

A view maps model points to paper: pick two model axes (or give a projection
direction), scale them, flip if needed, and offset to the view centre. Shapes
come in as loops: (outer, holes), each a list of 2D or 3D points.
"""
import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from .sheet import CHAIN, DASHED, NARROW, WIDE, lw

AXES = {'x': 0, 'y': 1, 'z': 2}


def signed_area(loop):
    if len(loop) < 3:
        return 0.0
    p = np.asarray(loop, float)[:, :2]
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compound_path(loops):
    """One matplotlib Path for several (outer, holes) shapes: outers counter-
    clockwise, holes clockwise, so the nonzero rule leaves the holes empty."""
    verts, codes = [], []
    for outer, holes in loops or []:
        for ring, want_ccw in [(outer, True)] + [(h, False) for h in holes]:
            ring = [tuple(p[:2]) for p in ring]
            if len(ring) < 3:
                continue
            if (signed_area(ring) > 0) != want_ccw:
                ring = ring[::-1]
            verts += ring + [ring[0]]
            codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 1) + [Path.CLOSEPOLY]
    if not verts:
        return None
    return Path(verts, codes)


class View:
    """A view of the part on the sheet.

    centre    paper position (mm) of the model origin
    scale     paper mm per model mm (2.0 for 2:1)
    axes      which model axes run along paper x and y, e.g. ('y', 'z')
    flip      +1 / -1 per paper axis, to mirror a model axis
    clip      optional (cx, cy, r) paper circle, for detail views
    """

    def __init__(self, sheet, centre, scale=1.0, axes=('x', 'y'), flip=(1, 1), clip=None):
        self.sheet = sheet
        self.ax = sheet.ax
        self.centre = centre
        self.scale = float(scale)
        self.axes = tuple(AXES[a] if isinstance(a, str) else a for a in axes)
        self.flip = flip
        self.clip = None
        if clip is not None:
            from matplotlib.patches import Circle
            if not clip[2] > 0:
                raise ValueError('a detail view needs a clip circle with a positive radius')
            self.clip = Circle(clip[:2], clip[2], transform=self.ax.transData)

    # ------------------------------------------------------------- mapping
    def to_paper(self, p):
        """Model point (2D or 3D) -> paper (x, y). 2D points use (x, y) directly."""
        if len(p) == 2:
            a, b = p[0], p[1]
        else:
            a, b = p[self.axes[0]], p[self.axes[1]]
        return (self.centre[0] + self.flip[0] * a * self.scale,
                self.centre[1] + self.flip[1] * b * self.scale)

    def model_length(self, paper_length):
        return paper_length / self.scale

    def _paper_loops(self, loops):
        return [([self.to_paper(p) for p in outer], [[self.to_paper(p) for p in h] for h in holes])
                for outer, holes in loops]

    def _add(self, artist):
        if self.clip is not None:
            artist.set_clip_path(self.clip)
        return artist

    # ------------------------------------------------------------- geometry
    def outline(self, loops, width=WIDE, fill='white'):
        """Visible outline of one or more (outer, holes) shapes. None if there is nothing to draw."""
        path = compound_path(self._paper_loops(loops))
        if path is None:
            return None
        return self._add(self.ax.add_patch(PathPatch(path, fc=fill, ec='k', lw=lw(width), zorder=3)))

    def section(self, loops, spacing=2.5, angle=45.0, width=WIDE):
        """Cut face: white fill, 45 deg hatching clipped to the exact face, wide outline.
        Spacing is on paper (ISO 128-50 suggests at least 0.7 mm; 2-3 mm reads well).
        None if there is nothing to draw (a plane that missed the part, say)."""
        paper = self._paper_loops(loops)
        path = compound_path(paper)
        if path is None:
            return None
        fill_path = path
        if self.clip is not None:
            # a patch takes one clip path, so cut the face to the detail circle
            # geometrically and hatch that; (A - H) & C == (A & C) - (H & C)
            window = circle_polygon(self.clip.center, self.clip.radius)
            cut = []
            for outer, holes in paper:
                o = clip_convex(outer, window)
                if len(o) >= 3:
                    cut.append((o, [h2 for h2 in (clip_convex(h, window) for h in holes) if len(h2) >= 3]))
            fill_path = compound_path(cut)
            if fill_path is None:
                return None
            paper = cut
        self.ax.add_patch(PathPatch(fill_path, fc='white', ec='none', zorder=3))
        pts = np.array([p for outer, holes in paper for ring in [outer] + holes for p in ring])
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        a = math.radians(angle)
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        half = math.hypot(x1 - x0, y1 - y0) / 2 + spacing
        segs = [[tuple(c + k * n - half * d), tuple(c + k * n + half * d)]
                for k in np.arange(-half, half + spacing, spacing)]
        hatch = LineCollection(segs, colors='k', linewidths=lw(NARROW * 0.7), zorder=4)
        hatch.set_clip_path(PathPatch(fill_path, transform=self.ax.transData))
        self.ax.add_collection(hatch)
        # outline of the real face; in a detail view it is clipped to the circle
        return self._add(self.ax.add_patch(PathPatch(path, fc='none', ec='k', lw=lw(width), zorder=5)))

    def lines(self, polylines, width=NARROW, style='-'):
        out = []
        for pl in polylines:
            xs, ys = zip(*[self.to_paper(p) for p in pl])
            out.append(self._add(self.ax.plot(xs, ys, color='k', lw=lw(width), ls=style, zorder=5)[0]))
        return out

    def hidden(self, polylines):
        return self.lines(polylines, width=NARROW, style=DASHED)

    def centre_lines(self, centre=(0, 0), half=10.0, horizontal=True, vertical=True):
        """Chain lines through a model point, extending 'half' model mm each way."""
        cx, cy = centre[:2] if len(centre) == 2 else (centre[self.axes[0]], centre[self.axes[1]])
        out = []
        if horizontal:
            out += self.lines([[(cx - half, cy), (cx + half, cy)]], width=0.18, style=CHAIN)
        if vertical:
            out += self.lines([[(cx, cy - half), (cx, cy + half)]], width=0.18, style=CHAIN)
        return out

    def centre_mark(self, centre, r):
        """Centre lines for a circle of model radius r, overhanging by 3 mm on paper."""
        return self.centre_lines(centre, r + 3.0 / self.scale)

    # ------------------------------------------------------------- annotation
    def title(self, text, offset=8.0, below=True, extent=None):
        """View name ('SECTION A-A  2:1'), centred under (or over) the view."""
        cx, cy = self.centre
        y = cy - (extent or 0) - offset if below else cy + (extent or 0) + offset
        return self.sheet.text(cx, y, text, cap=3.5, ha='center', weight='bold')

    def cutting_plane(self, a, b, letter, look=(0, -1)):
        """Section line from model a to b: wide ends, thin chain between, arrows
        pointing in the viewing direction 'look' (paper vector), letters at each end.

        The arrows must match the section: mesh.section(normal=n) is viewed from the
        +n side looking along -n (u right, v up), so 'look' is -n as seen in this view."""
        pa, pb = np.array(self.to_paper(a)), np.array(self.to_paper(b))
        if np.linalg.norm(pb - pa) < 1e-9 or np.linalg.norm(look) < 1e-12:
            raise ValueError('a cutting plane needs two distinct ends and a non-zero look direction')
        u = (pb - pa) / np.linalg.norm(pb - pa)
        self.sheet.line([tuple(pa), tuple(pb)], width=0.18, style=CHAIN)
        look = np.array(look, float) / np.linalg.norm(look)
        for end, sgn in ((pa, 1), (pb, -1)):
            self.sheet.line([tuple(end), tuple(end + sgn * 6 * u)], width=0.7)
            tail = end + sgn * 3 * u
            arrow(self.sheet, tuple(tail), tuple(tail + 7 * look))
            # letter just beyond the arrowhead, centred on it whichever way it points
            self.sheet.text(*(tail + 12.0 * look), letter, cap=5.0, ha='center', va='center',
                            weight='bold')

    def detail_marker(self, centre, r, letter):
        """Thin circle round the region a detail view enlarges, with its letter."""
        c = self.to_paper(centre)
        R = r * self.scale
        self.sheet.circle(c, R, width=NARROW, style=DASHED)
        self.sheet.text(c[0] + R * 0.75, c[1] + R * 0.75, letter, cap=5.0, weight='bold')


def circle_polygon(c, r, n=96):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a in t]


def clip_convex(ring, window):
    """Sutherland-Hodgman: clip a ring (any shape) against a convex counter-clockwise
    window. A concave ring can come back with zero-width bridges along the window
    edge; they have no area, so fill and hatching are unaffected."""
    if len(window) < 3 or signed_area(window) <= 1e-12:
        return []                                   # an empty window keeps nothing
    out = [tuple(p[:2]) for p in ring]
    m = len(window)
    for i in range(m):
        a, b = window[i], window[(i + 1) % m]

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def cross(p, q):
            x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
            x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        src, out = out, []
        if not src:
            break
        prev = src[-1]
        for cur in src:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
    return out


def arrow(sheet, tail, tip, length=3.0, width=2 * 3.0 * math.tan(math.radians(15))):
    """Closed filled arrowhead at tip (ISO 129-1: about 3 mm long, 30 deg included angle)."""
    t, h = np.array(tail, float), np.array(tip, float)
    u = h - t
    n = np.linalg.norm(u)
    if n == 0:
        return
    u /= n
    v = np.array([-u[1], u[0]])
    base = h - length * u
    sheet.line([tuple(t), tuple(base)], width=NARROW)
    sheet.polygon([tuple(h), tuple(base + width / 2 * v), tuple(base - width / 2 * v)], width=0.1, fill='k')
