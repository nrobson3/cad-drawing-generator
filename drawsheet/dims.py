"""Dimensions (ISO 129-1 style): extension lines, filled arrowheads, values above
the dimension line, read from the bottom or the right of the sheet.

Every function takes model points through a View, so the value it prints is the
true model size whatever the view scale.
"""
import math
import numbers

import numpy as np

from .sheet import NARROW, text_width
from .view import arrow

GAP = 1.0          # paper gap between the feature and the extension line
OVERSHOOT = 2.0    # extension line beyond the dimension line


def fmt(value, places=2):
    """Model value as drawn: trailing zeros after the decimal point dropped (12.50 -> 12.5,
    40.0 -> 40); integer digits are never touched."""
    s = '%.*f' % (places, value)
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def tolerance_text(value, tol=None, places=2):
    """'12.5', '12.5 ±0.05', or deviations '12 +0.018/0' from an (upper, lower) pair."""
    base = fmt(value, places)
    if tol is None:
        return base
    if isinstance(tol, numbers.Real):          # numpy scalars included
        if tol < 0:
            raise ValueError('a ± tolerance must be positive; give (upper, lower) for deviations')
        return '%s ±%s' % (base, fmt(tol, 3))
    upper, lower = tol            # signed deviations, e.g. (+0.018, 0) for H7, (-0.007, -0.020) for g6
    # ISO 129-1: both deviations to the same number of decimals (-0.007/-0.020, not -0.02)
    dp = max(len(fmt(abs(x), 3).partition('.')[2]) for x in (upper, lower))

    def dev(x):
        s = '%.*f' % (dp, abs(x))
        return '0' if float(s) == 0 else ('+' if x > 0 else '-') + s
    return '%s %s/%s' % (base, dev(upper), dev(lower))


def linear(view, a, b, offset, direction='h', text=None, tol=None, basic=False, places=2, cap=3.5):
    """Linear dimension between model points a and b.

    direction  'h' horizontal, 'v' vertical (paper), or 'a' aligned with a-b
    offset     paper distance from the points to the dimension line: positive is
               up for 'h', right for 'v', and to the left of a->b for 'a'; 0 puts the
               dimension line through the points (inside a bore, say)
    text       override the measured value; tol adds ±t or a (plus, minus) pair
    basic      box the value (a theoretically exact dimension, ISO 1101)
    """
    sheet = view.sheet
    pa, pb = np.array(view.to_paper(a)), np.array(view.to_paper(b))
    if np.linalg.norm(pb - pa) < 1e-9:
        raise ValueError('a linear dimension needs two distinct points')
    if direction == 'h':
        u, n = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    elif direction == 'v':
        u, n = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    else:
        u = (pb - pa) / np.linalg.norm(pb - pa)
        n = np.array([-u[1], u[0]])
    # dimension line position: offset from the outermost point along n
    base = max(np.dot(pa, n), np.dot(pb, n)) if offset >= 0 else min(np.dot(pa, n), np.dot(pb, n))
    level = base + offset
    da = pa + (level - np.dot(pa, n)) * n
    db = pb + (level - np.dot(pb, n)) * n
    s = 1.0 if offset >= 0 else -1.0
    if offset != 0:                            # no extension lines when the line runs through the points
        for p, d in ((pa, da), (pb, db)):
            sheet.line([tuple(p + s * GAP * n), tuple(d + s * OVERSHOOT * n)], width=NARROW)
    measured = abs(np.dot(pb - pa, u)) / view.scale
    label = text if text is not None else tolerance_text(measured, tol, places)
    mid = (da + db) / 2
    length = np.linalg.norm(db - da)
    if length > 10:                            # arrows inside
        arrow(sheet, tuple(mid), tuple(da))
        arrow(sheet, tuple(mid), tuple(db))
    else:                                      # too tight: arrows outside, pointing in
        w = (db - da) / (length or 1)
        arrow(sheet, tuple(da - 7 * w), tuple(da))
        arrow(sheet, tuple(db + 7 * w), tuple(db))
        sheet.line([tuple(da), tuple(db)], width=NARROW)
    angle = math.degrees(math.atan2(u[1], u[0]))
    if angle > 90.0 or angle <= -90.0:         # keep text readable from the bottom / right
        angle -= 180.0 if angle > 0 else -180.0
    up = np.array([-math.sin(math.radians(angle)), math.cos(math.radians(angle))])
    tp = mid + 1.2 * up
    box = dict(boxstyle='square,pad=0.25', fc='white', ec='k', lw=0.6) if basic else None
    sheet.text(tp[0], tp[1], label, cap=cap, ha='center', va='bottom', rotation=angle,
               rotation_mode='anchor', bbox=box, zorder=7)
    return measured


def diameter(view, centre, r, angle=45.0, text=None, tol=None, places=2, count=None, cap=3.5):
    """Diameter across a circle of model radius r, line at 'angle' degrees on paper."""
    sheet = view.sheet
    c = np.array(view.to_paper(centre))
    R = r * view.scale
    u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    p1, p2 = c - R * u, c + R * u
    arrow(sheet, tuple(c), tuple(p1))
    arrow(sheet, tuple(c), tuple(p2))
    tail = p2 + 8 * u
    sheet.line([tuple(p2), tuple(tail)], width=NARROW)
    label = text if text is not None else '%s%s%s' % ('%dx ' % count if count else '', chr(0x2300),
                                                       tolerance_text(2 * r, tol, places))
    shelf = 1 if u[0] >= 0 else -1
    end = tail + np.array([shelf * (text_width(label, cap) + 2), 0])
    sheet.line([tuple(tail), tuple(end)], width=NARROW)
    sheet.text(tail[0] + shelf * 1.0, tail[1] + 1.0, label, cap=cap, ha='left' if shelf > 0 else 'right')
    return 2 * r


def radius(view, centre, r, angle=45.0, text=None, tol=None, places=2, cap=3.5):
    """Radius: line from the centre side to the arc, one arrow on the arc."""
    sheet = view.sheet
    c = np.array(view.to_paper(centre))
    R = r * view.scale
    u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    p = c + R * u
    start = c + max(0.0, R - 12) * u
    arrow(sheet, tuple(start), tuple(p))
    label = text if text is not None else 'R' + tolerance_text(r, tol, places)
    tail = p + 6 * u
    sheet.line([tuple(p), tuple(tail)], width=NARROW)
    shelf = 1 if u[0] >= 0 else -1
    sheet.line([tuple(tail), tuple(tail + np.array([shelf * (text_width(label, cap) + 2), 0]))], width=NARROW)
    sheet.text(tail[0] + shelf * 1.0, tail[1] + 1.0, label, cap=cap, ha='left' if shelf > 0 else 'right')
    return r


def leader(view, point, text_at, text, cap=3.5, dot=True):
    """Leader from a model point to a note at paper position text_at."""
    sheet = view.sheet
    p = view.to_paper(point)
    sheet.line([p, text_at], width=NARROW)
    if dot:
        sheet.circle(p, 0.5, width=0.1, fill='k')
    shelf = 1 if text_at[0] >= p[0] else -1
    width = max(text_width(s, cap) for s in text.split('\n')) + 2
    sheet.line([text_at, (text_at[0] + shelf * width, text_at[1])], width=NARROW)
    sheet.text(text_at[0] + shelf * 1.0, text_at[1] + 1.0, text, cap=cap, ha='left' if shelf > 0 else 'right')
