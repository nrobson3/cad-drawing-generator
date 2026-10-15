"""Sections from a triangle mesh, so any part exported as STL can be drawn.

section(tris, origin, normal) cuts the mesh with a plane and returns the cut
face as (outer, holes) loops in plane coordinates, ready for View.section().
Export the STL at a fine refinement: curved edges come out as straight chords
of the mesh, so chord height on the mesh is chord height on the drawing.

The section is viewed from the +normal side looking along -normal, with the
plane's u axis to the right and v up. Do not cut exactly on a face of the part
(a floor at z = 2, say): offset the plane a little into the material, as the
examples do, or the cut may come back empty.
"""
import struct
import warnings

import numpy as np


def read_stl(path):
    """Triangles as an (n, 3, 3) float array. ASCII or binary."""
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack('<I', data[80:84])[0]
        if 84 + 50 * n <= len(data) and not data[:5].lower() == b'solid' or 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]),
                                count=n, offset=84)
            return rec['v'].astype(float)
    verts = [list(map(float, line.split()[1:4])) for line in data.decode('ascii', 'replace').splitlines()
             if line.strip().lower().startswith('vertex')]
    if not verts or len(verts) % 3:
        raise ValueError('%s: no triangles read (not a valid ASCII or binary STL)' % path)
    return np.array(verts, float).reshape(-1, 3, 3)


def plane_basis(normal, up=None):
    """Two in-plane unit axes (u, v) for a plane normal, so (u, v, n) is right-handed."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    if up is None:
        up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.asarray(up, float) - np.dot(up, n) * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    return u, v, n


def _segments(tris, origin, n, eps):
    """Plane/triangle intersection segments (3D), each directed so the material is on
    its left seen from the +n side: with the triangle's outward normal N (from its
    winding), (b - a) x n points along N."""
    d = np.einsum('tij,j->ti', tris - origin, n)          # signed distances of each vertex
    d[np.abs(d) < eps] = eps                              # nudge vertices that sit on the plane, consistently
    mask = (d.min(1) < 0) & (d.max(1) > 0)
    segs = []
    for tri, dist in zip(tris[mask], d[mask]):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dist[i] < 0) != (dist[j] < 0):
                t = dist[i] / (dist[i] - dist[j])
                pts.append(tri[i] + t * (tri[j] - tri[i]))
        if len(pts) == 2:
            a, b = pts
            N = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if np.dot(np.cross(b - a, n), N) < 0:
                a, b = b, a
            segs.append((a, b))
    return segs


def _chain(segs2d, tol):
    """Join directed segments head to tail into loops.

    Endpoints closer than tol are first merged into one node (union-find over a grid
    of tol-sized cells, neighbouring cells included). A segment whose two ends merge
    is dropped: it is a sliver where the plane passes through a mesh vertex. On a
    closed mesh every node then has as many segments leaving as arriving, so each
    walk returns to its start. Returns (loops, open_count), where open_count is the
    number of chains that start at a node with more segments out than in: gaps."""
    pts = [p for seg in segs2d for p in seg]
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grid = {}
    for i, p in enumerate(pts):
        cx, cy = int(np.floor(p[0] / tol)), int(np.floor(p[1] / tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    q = pts[j]
                    if abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol:
                        parent[find(i)] = find(j)
        grid.setdefault((cx, cy), []).append(i)
    node = [find(i) for i in range(len(pts))]

    out, balance = {}, {}
    for k in range(len(segs2d)):
        a, b = node[2 * k], node[2 * k + 1]
        if a != b:
            out.setdefault(a, []).append(b)
            balance[a] = balance.get(a, 0) + 1
            balance[b] = balance.get(b, 0) - 1
    open_count = sum(v for v in balance.values() if v > 0)

    loops = []
    for start in list(out):
        while out.get(start):
            loop, cur = [start], out[start].pop()
            while cur != start and out.get(cur):
                loop.append(cur)
                cur = out[cur].pop()
            if cur == start and len(loop) >= 3:
                loops.append([pts[i] for i in loop])
    return loops, open_count


def _area(loop):
    p = np.asarray(loop)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _inside(pt, loop):
    """Even-odd point-in-polygon."""
    x, y = pt
    p = np.asarray(loop)
    xi, yi = p[:, 0], p[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
    return bool(np.count_nonzero(cross) % 2)


def nest(loops):
    """Group loops into (outer, holes): a loop inside an even number of others is an
    outer boundary, inside an odd number it is a hole of its immediate parent."""
    loops = sorted(loops, key=lambda l: -abs(_area(l)))
    depth, parent = [], []
    for i, l in enumerate(loops):
        probe = l[0]
        containers = [j for j in range(i) if _inside(probe, loops[j])]
        depth.append(len(containers))
        parent.append(containers[-1] if containers else None)
    shapes = {}
    for i, l in enumerate(loops):
        if depth[i] % 2 == 0:
            shapes[i] = (l, [])
    for i, l in enumerate(loops):
        if depth[i] % 2 == 1 and parent[i] in shapes:
            shapes[parent[i]][1].append(l)
    return list(shapes.values())


def section(tris, origin, normal, up=None, tol=None):
    """Cut face of the mesh on a plane, as [(outer, holes), ...] in plane (u, v)
    coordinates measured from 'origin'. 'up' picks the plane's v axis. Tolerances
    scale with the part (STL coordinates are 32-bit floats). Warns if a chain of
    cut segments fails to close, which means the mesh is not watertight there."""
    tris = np.asarray(tris, float)
    if len(tris) == 0:
        raise ValueError('empty mesh')
    origin = np.asarray(origin, float)
    size = float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0))) or 1.0
    tol = tol or 1e-6 * size
    u, v, n = plane_basis(normal, up)
    segs = _segments(tris, origin, n, 1e-12 * size)
    seg2 = [((float(np.dot(a - origin, u)), float(np.dot(a - origin, v))),
             (float(np.dot(b - origin, u)), float(np.dot(b - origin, v)))) for a, b in segs]
    loops, open_count = _chain(seg2, tol)
    if open_count:
        warnings.warn('%d open chain(s) in the section: the mesh is not closed there' % open_count)
    return nest(loops)


def area(shapes):
    """Net area of [(outer, holes)] shapes."""
    return sum(abs(_area(o)) - sum(abs(_area(h)) for h in holes) for o, holes in shapes)
