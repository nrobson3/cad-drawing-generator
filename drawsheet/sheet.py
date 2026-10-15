"""Drawing sheet: frame, zones, title block, projection symbol, notes.

Coordinates are millimetres on the paper, origin at the bottom-left corner of
the trimmed sheet. Line widths follow ISO 128-2 line group 0.5 (wide 0.5 mm,
narrow 0.25 mm); the frame follows ISO 5457 (20 mm filing margin on the left,
10 mm elsewhere, 20 mm all round on A1; zones about 50 mm long); the title block carries the ISO 7200
fields.
"""
import os

from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.textpath import TextPath

SIZES = {'A4': (297.0, 210.0), 'A3': (420.0, 297.0), 'A2': (594.0, 420.0), 'A1': (841.0, 594.0)}
PT_PER_MM = 72.0 / 25.4
WIDE = 0.5            # mm, visible outlines, frame
NARROW = 0.25         # mm, dimensions, hatching, centre lines
CHAIN = (0, (12, 3, 2, 3))    # long-dash-dot, in units of line width
DASHED = (0, (6, 3))

# ISO 3098-style lettering if a matching font is installed, otherwise DejaVu Sans
_FONTS = ('osifont', 'DS ISO 1', 'ISOCPEUR', 'DejaVu Sans')
FONT = next(f for f in _FONTS if f == 'DejaVu Sans' or any(e.name == f for e in font_manager.fontManager.ttflist))


def lw(mm):
    """Line width in points for a width in millimetres."""
    return mm * PT_PER_MM


def font_pt(cap_mm):
    """Font size in points for a capital height in millimetres (caps are ~0.72 em)."""
    return cap_mm * PT_PER_MM / 0.72


def text_width(s, cap):
    """Rendered width (mm) of a string at a capital height, measured in the sheet's font."""
    if not s:
        return 0.0
    path = TextPath((0, 0), s.replace(' ', chr(0xA0)), size=font_pt(cap) / PT_PER_MM,
                    prop=FontProperties(family=FONT))
    return float(path.get_extents().x1)


def zone_count(length):
    """ISO 5457: an even number of zones, each close to 50 mm long."""
    n = max(2, int(round(length / 50.0)))
    return n + (n % 2)


class Sheet:
    """One drawing sheet. Draw on it through views, dims and gdt, then save()."""

    def __init__(self, size='A3', projection='first', title='', number='', rev='A', scale='1:1',
                 material='', finish='', mass='', drawn='', date='', units='mm',
                 tolerance='ISO 2768-mK', sheet='1 / 1', owner=''):
        if size not in SIZES:
            raise ValueError('size must be one of %s' % ', '.join(SIZES))
        if projection not in ('first', 'third'):
            raise ValueError("projection must be 'first' or 'third'")
        self.size = size
        self.w, self.h = SIZES[size]
        self.projection = projection
        self.fields = dict(title=title, number=number, rev=rev, scale=scale, material=material,
                           finish=finish, mass=mass, drawn=drawn, date=date, units=units,
                           tolerance=tolerance, sheet=sheet, owner=owner)
        self.fields = {k: '' if v is None else str(v) for k, v in self.fields.items()}
        # a bare Figure, so importing drawsheet never changes the host's matplotlib backend
        self.fig = Figure(figsize=(self.w / 25.4, self.h / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        # drawing area inside the frame: 20 mm filing margin on the left; the other
        # margins are 10 mm on A2-A4 and 20 mm on A1 (ISO 5457)
        m = 20.0 if size == 'A1' else 10.0
        self.left, self.bottom = 20.0, m
        self.right, self.top = self.w - m, self.h - m
        self._frame()
        self.title_block = self._title_block()

    # ------------------------------------------------------------- primitives
    def text(self, x, y, s, cap=3.5, ha='left', va='baseline', rotation=0, **kw):
        kw.setdefault('family', FONT)
        kw.setdefault('color', 'k')
        return self.ax.text(x, y, s, fontsize=font_pt(cap), ha=ha, va=va, rotation=rotation, **kw)

    def line(self, pts, width=NARROW, style='-', color='k', zorder=3):
        xs, ys = zip(*pts)
        return self.ax.plot(xs, ys, color=color, lw=lw(width), ls=style, solid_capstyle='butt',
                            dash_capstyle='butt', zorder=zorder)[0]

    def rect(self, x, y, w, h, width=NARROW, fill='none', zorder=3):
        return self.ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec='k', lw=lw(width), zorder=zorder))

    def polygon(self, pts, width=NARROW, fill='none', edge='k', zorder=3):
        return self.ax.add_patch(Polygon(pts, closed=True, fc=fill, ec=edge, lw=lw(width), zorder=zorder))

    def circle(self, c, r, width=NARROW, fill='none', style='-', zorder=3):
        return self.ax.add_patch(Circle(c, r, fc=fill, ec='k', lw=lw(width), ls=style, zorder=zorder))

    # ------------------------------------------------------------- frame
    def _frame(self):
        self.rect(self.left, self.bottom, self.right - self.left, self.top - self.bottom, width=0.7)
        nx, ny = zone_count(self.right - self.left), zone_count(self.top - self.bottom)
        zx = (self.right - self.left) / nx
        zy = (self.top - self.bottom) / ny
        for i in range(nx + 1):                      # zone ticks and numbers, top and bottom
            x = self.left + i * zx
            for y0, y1 in ((self.bottom - 5, self.bottom), (self.top, self.top + 5)):
                self.line([(x, y0), (x, y1)], width=NARROW)
            if i < nx:
                for y in (self.bottom - 3.6, self.top + 1.6):
                    self.text(x + zx / 2, y, str(i + 1), cap=2.5, ha='center')
        letters = 'ABCDEFGHJKLMNPRSTUVWXYZ'          # ISO 5457 skips I, O and Q
        for j in range(ny + 1):                      # zone ticks and letters, left and right
            y = self.bottom + j * zy
            for x0, x1 in ((self.left - 5, self.left), (self.right, self.right + 5)):
                self.line([(x0, y), (x1, y)], width=NARROW)
            if j < ny:
                for x in (self.left - 2.5, self.right + 2.5):
                    self.text(x, self.top - (j + 0.5) * zy, letters[j], cap=2.5, ha='center', va='center')
        # centring marks at the middle of each edge
        mx, my = self.w / 2, self.h / 2
        for pts in (((mx, 0), (mx, self.bottom + 5)), ((mx, self.h), (mx, self.top - 5)),
                    ((0, my), (self.left + 5, my)), ((self.w, my), (self.right - 5, my))):
            self.line(pts, width=0.7)
        self.zones = (nx, ny)

    # ------------------------------------------------------------- title block
    def _title_block(self):
        f = self.fields
        w, h = 180.0, 52.0
        x0, y0 = self.right - w, self.bottom
        self.rect(x0, y0, w, h, width=WIDE, fill='white', zorder=4)
        rows = [y0 + 12, y0 + 24, y0 + 36]
        for y in rows:
            self.line([(x0, y), (x0 + w, y)], zorder=5)
        for x, ya, yb in ((x0 + 60, y0, y0 + 36), (x0 + 120, y0, y0 + 36), (x0 + 150, y0 + 12, y0 + 36)):
            self.line([(x, ya), (x, yb)], zorder=5)
        self.text(x0 + 2, y0 + h - 7, f['title'], cap=5.0, weight='bold', zorder=6)
        if f['owner']:
            self.text(x0 + w - 2, y0 + h - 7, f['owner'], cap=2.5, ha='right', zorder=6)
        cells = [
            (x0 + 2, y0 + 30, 'MATERIAL', f['material']), (x0 + 62, y0 + 30, 'FINISH', f['finish']),
            (x0 + 122, y0 + 30, 'MASS', f['mass']), (x0 + 152, y0 + 30, 'SHEET', f['sheet']),
            (x0 + 2, y0 + 18, 'DRAWN', ('%s  %s' % (f['drawn'], f['date'])).strip()),
            (x0 + 62, y0 + 18, 'GENERAL TOLERANCES', f['tolerance']),
            (x0 + 122, y0 + 18, 'SCALE', f['scale']), (x0 + 152, y0 + 18, 'UNITS', f['units']),
            (x0 + 62, y0 + 6, 'DWG NO', f['number']), (x0 + 122, y0 + 6, 'REV', f['rev']),
        ]
        for x, y, key, value in cells:
            self.text(x, y + 2.8, key, cap=1.8, color='#444444', zorder=6)
            self.text(x, y - 2.2, value, cap=2.5, zorder=6)
        self.text(x0 + 2, y0 + 8.8, 'PROJECTION', cap=1.8, color='#444444', zorder=6)
        projection_symbol(self, x0 + 30, y0 + 4.5, self.projection, size=1.0)
        return (x0, y0, w, h)

    def notes(self, lines, x=None, y=None, cap=2.5, heading='NOTES'):
        """Numbered notes, top-down from (x, y); defaults to the lower left of the
        frame, raised as far as a long list needs to stay inside it."""
        x = self.left + 5 if x is None else x
        if y is None:
            y = max(self.bottom + 50, self.bottom + 4 + 2 * cap + (len(lines) - 1) * 2.1 * cap)
        self.text(x, y, heading, cap=3.5, weight='bold')
        for i, n in enumerate(lines, 1):
            self.text(x, y - 2 * cap - (i - 1) * 2.1 * cap, '%d.  %s' % (i, n), cap=cap)

    def save(self, path, dpi=200):
        """Write .pdf, .svg or .png (by extension). PDF and SVG stay vector."""
        if os.path.splitext(str(path))[1].lower() not in ('.pdf', '.svg', '.png'):
            raise ValueError('save to a .pdf, .svg or .png path')
        self.fig.savefig(path, dpi=dpi)
        return path

    def close(self):
        """Kept for symmetry; a bare Figure needs no closing."""
        self.fig.clear()


def projection_symbol(sheet, x, y, method='first', size=1.0):
    """ISO 128 / ISO 5456-2 projection symbol, its centre line on y, starting at x.
    First angle: the truncated cone (side view) sits left of the circles (end view);
    third angle swaps the two. Both keep the cone's narrow end on the left."""
    s = size
    cone_x = x if method == 'first' else x + 12.5 * s
    circ_x = x + 16 * s if method == 'first' else x + 3.5 * s
    sheet.polygon([(cone_x, y - 1.75 * s), (cone_x + 7 * s, y - 3.5 * s),
                   (cone_x + 7 * s, y + 3.5 * s), (cone_x, y + 1.75 * s)], zorder=6)
    sheet.circle((circ_x, y), 3.5 * s, zorder=6)
    sheet.circle((circ_x, y), 1.75 * s, zorder=6)
    sheet.line([(x - 2 * s, y), (x + 22 * s, y)], style=CHAIN, width=0.18, zorder=6)
