"""Geometric tolerancing (ISO 1101 / ASME Y14.5 symbols).

The 14 characteristic symbols are drawn as lines rather than taken from a font,
so they look the same on every machine and in every export format.
"""
import math

import numpy as np

from .sheet import NARROW, text_width

CHARACTERISTICS = {
    # form
    'straightness', 'flatness', 'circularity', 'cylindricity',
    # profile
    'profile_line', 'profile_surface',
    # orientation
    'parallelism', 'perpendicularity', 'angularity',
    # location
    'position', 'concentricity', 'symmetry',
    # runout
    'circular_runout', 'total_runout',
}
# orientation, runout, concentricity and symmetry are always relative to a datum;
# position may stand alone (a hole pattern located only to itself)
NEEDS_DATUM = {'parallelism', 'perpendicularity', 'angularity', 'concentricity', 'symmetry',
               'circular_runout', 'total_runout'}


def symbol(sheet, kind, cx, cy, s):
    """Draw one characteristic symbol centred on (cx, cy); s is the frame height."""
    k = s * 0.32          # half-size of the glyph
    w = NARROW
    L = sheet.line

    def arc(c, r, a0, a1, n=24):
        t = np.linspace(math.radians(a0), math.radians(a1), n)
        return [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a in t]

    def arrowhead(tip, d):
        d = np.array(d, float) / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        base = np.array(tip) - 0.35 * s * d
        sheet.polygon([tip, tuple(base + 0.12 * s * n), tuple(base - 0.12 * s * n)], width=0.1, fill='k', zorder=8)

    if kind == 'straightness':
        L([(cx - k, cy), (cx + k, cy)], width=w, zorder=8)
    elif kind == 'flatness':
        L([(cx - k, cy - 0.45 * k), (cx + 0.5 * k, cy - 0.45 * k), (cx + k, cy + 0.45 * k),
           (cx - 0.5 * k, cy + 0.45 * k), (cx - k, cy - 0.45 * k)], width=w, zorder=8)
    elif kind == 'circularity':
        sheet.circle((cx, cy), k * 0.8, width=w, zorder=8)
    elif kind == 'cylindricity':
        sheet.circle((cx, cy), k * 0.65, width=w, zorder=8)
        L([(cx - k * 0.75, cy - k), (cx - k * 0.1, cy + k)], width=w, zorder=8)
        L([(cx + k * 0.1, cy - k), (cx + k * 0.75, cy + k)], width=w, zorder=8)
    elif kind == 'profile_line':
        L(arc((cx, cy - 0.5 * k), k, 0, 180), width=w, zorder=8)
    elif kind == 'profile_surface':
        pts = arc((cx, cy - 0.5 * k), k, 0, 180)
        L(pts + [pts[0]], width=w, zorder=8)
    elif kind == 'parallelism':
        for dx in (-0.35 * k, 0.35 * k):
            L([(cx + dx - 0.4 * k, cy - k), (cx + dx + 0.4 * k, cy + k)], width=w, zorder=8)
    elif kind == 'perpendicularity':
        L([(cx - k, cy - k), (cx + k, cy - k)], width=w, zorder=8)
        L([(cx, cy - k), (cx, cy + k)], width=w, zorder=8)
    elif kind == 'angularity':
        L([(cx + k, cy - k), (cx - k, cy - k), (cx + k * 0.8, cy + k * 0.6)], width=w, zorder=8)
    elif kind == 'position':
        sheet.circle((cx, cy), k * 0.6, width=w, zorder=8)
        L([(cx - k, cy), (cx + k, cy)], width=w, zorder=8)
        L([(cx, cy - k), (cx, cy + k)], width=w, zorder=8)
    elif kind == 'concentricity':
        sheet.circle((cx, cy), k * 0.85, width=w, zorder=8)
        sheet.circle((cx, cy), k * 0.45, width=w, zorder=8)
    elif kind == 'symmetry':
        L([(cx - k, cy), (cx + k, cy)], width=w, zorder=8)
        L([(cx - 0.65 * k, cy + 0.55 * k), (cx + 0.65 * k, cy + 0.55 * k)], width=w, zorder=8)
        L([(cx - 0.65 * k, cy - 0.55 * k), (cx + 0.65 * k, cy - 0.55 * k)], width=w, zorder=8)
    elif kind == 'circular_runout':
        L([(cx - 0.7 * k, cy - k), (cx + 0.5 * k, cy + 0.6 * k)], width=w, zorder=8)
        arrowhead((cx + 0.7 * k, cy + 0.9 * k), (0.6, 0.8))
    elif kind == 'total_runout':
        for dx in (-0.45 * k, 0.35 * k):
            L([(cx + dx - 0.45 * k, cy - k), (cx + dx + 0.4 * k, cy + 0.55 * k)], width=w, zorder=8)
            arrowhead((cx + dx + 0.6 * k, cy + 0.9 * k), (0.5, 0.86))
        L([(cx - 0.9 * k, cy - k), (cx + 0.75 * k, cy - k)], width=w, zorder=8)
    else:
        raise ValueError('unknown characteristic %r' % kind)


def feature_control_frame(sheet, x, y, kind, tolerance, datums=(), diameter=False, modifier=None, cap=3.5):
    """Feature control frame with its lower-left corner at paper (x, y).

    tolerance  a number or string ('0.1')
    datums     datum letters in order of precedence, e.g. ('A', 'B', 'C')
    diameter   prefix the tolerance with the diameter sign (cylindrical zone)
    modifier   'M' or 'L' (maximum / least material condition), drawn circled
    Returns the paper x of the frame's right end.
    """
    if kind not in CHARACTERISTICS:
        raise ValueError('unknown characteristic %r' % kind)
    if kind in NEEDS_DATUM and not datums:
        raise ValueError('%s needs at least one datum' % kind)
    h = 2.0 * cap                                   # ISO 1101: frame height twice the lettering
    cells = [('symbol', h)]
    tol = ('%s%s' % (chr(0x2300) if diameter else '', tolerance)).strip()
    mod_w = 1.6 * cap if modifier else 0.0
    cells.append(('text', max(h, text_width(tol, cap) + mod_w + 2.4)))
    for d in datums:
        cells.append(('datum', max(h, text_width(d, cap) + 2.4)))
    cx = x
    for i, (what, w) in enumerate(cells):
        sheet.rect(cx, y, w, h, width=NARROW, fill='white', zorder=7)
        if what == 'symbol':
            symbol(sheet, kind, cx + w / 2, y + h / 2, h)
        elif what == 'text':
            sheet.text(cx + 1.2, y + h / 2, tol, cap=cap, va='center_baseline', zorder=8)
            if modifier:
                mx = cx + 1.2 + text_width(tol, cap) + 0.8 * cap
                sheet.circle((mx, y + h / 2), cap * 0.65, width=NARROW, zorder=8)
                sheet.text(mx, y + h / 2, modifier, cap=cap * 0.75, ha='center', va='center_baseline', zorder=8)
        else:
            sheet.text(cx + w / 2, y + h / 2, datums[i - 2], cap=cap, ha='center', va='center_baseline', zorder=8)
        cx += w
    return cx


def datum_feature(sheet, p, letter, direction=(0, -1), length=7.0, cap=3.5):
    """Datum feature symbol: filled triangle on the feature at paper p, a stem in
    'direction', and the boxed letter at the end."""
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    p = np.array(p, float)
    tri = [tuple(p + 1.8 * n), tuple(p - 1.8 * n), tuple(p + 3.0 * d)]
    sheet.polygon(tri, width=0.1, fill='k', zorder=7)
    end = p + length * d
    sheet.line([tuple(p + 3.0 * d), tuple(end)], width=NARROW, zorder=7)
    b = 2.0 * cap
    box_c = end + d * b / 2
    sheet.rect(box_c[0] - b / 2, box_c[1] - b / 2, b, b, width=NARROW, fill='white', zorder=7)
    sheet.text(box_c[0], box_c[1], letter, cap=cap, ha='center', va='center_baseline', zorder=8)
