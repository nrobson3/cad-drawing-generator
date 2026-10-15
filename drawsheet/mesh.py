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
import math
import struct
import warnings

import numpy as np


def read_stl(path):
    """Triangles as an (n, 3, 3) float array. ASCII or binary.

    A binary file whose size matches its triangle count is read as binary even if
    its 80-byte header starts with 'solid' (some exporters write that); a binary
    file with padding after the triangles is read too."""
    with open(path, 'rb') as f:
        data = f.read()
    n = struct.unpack('<I', data[80:84])[0] if len(data) >= 84 else None

    def binary():
        rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]),
                            count=n, offset=84)
        return rec['v'].astype(float)

    if n is not None and 84 + 50 * n == len(data):
        return binary()
    verts = []
    if data.lstrip()[:5].lower() == b'solid':
        verts = [list(map(float, line.split()[1:4])) for line in data.decode('ascii', 'replace').splitlines()
                 if line.strip().lower().startswith('vertex')]
    if verts and len(verts) % 3 == 0:
        return np.array(verts, float).reshape(-1, 3, 3)
    if n and 84 + 50 * n <= len(data):
        return binary()
    raise ValueError('%s: no triangles read (not a valid ASCII or binary STL)' % path)


def plane_basis(normal, up=None):
    """Two in-plane unit axes (u, v) for a plane normal, so (u, v, n) is right-handed."""
    n = np.asarray(normal, float)
    if not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-12:
        raise ValueError('plane normal must be a non-zero vector')
    n = n / np.linalg.norm(n)
    if up is None:
        up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    up = np.asarray(up, float)
    v = up - np.dot(up, n) * n
    if np.linalg.norm(v) < 1e-9 * max(np.linalg.norm(up), 1e-300):
        raise ValueError("'up' is parallel to the plane normal; it must have a component in the plane")
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    return u, v, n


def _weld(tris, grid):
    """Shared vertices: coordinates equal to within 'grid' become one vertex."""
    flat = tris.reshape(-1, 3)
    q = np.round(flat / grid).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return flat[first], inv.reshape(-1, 3)


def _cut(verts, faces, origin, n, eps):
    """Cut segments of the plane through the welded mesh.

    A cut point lies on a mesh edge, so it is named by that edge (its two vertex
    numbers) and computed once: the two faces that share the edge then meet at
    exactly the same node, however close other cut points are. Vertices within
    eps of the plane count as above it, the same way for every face that uses
    them. Each segment is directed so the material is on its left seen from +n
    (from the face's winding: (b - a) x n points along the outward normal).
    Returns ([(node_a, node_b), ...], {node: 3D point})."""
    d = (verts - origin) @ n
    d[np.abs(d) < eps] = eps
    fd = d[faces]
    points, segs = {}, []
    for f in np.flatnonzero((fd.min(1) < 0) & (fd.max(1) > 0)):
        tri = faces[f]
        ends = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(tri[i]), int(tri[j])
            if (d[a] < 0) != (d[b] < 0):
                key = (a, b) if a < b else (b, a)
                if key not in points:
                    lo, hi = key
                    t = d[lo] / (d[lo] - d[hi])
                    points[key] = verts[lo] + t * (verts[hi] - verts[lo])
                ends.append(key)
        p, q = points[ends[0]], points[ends[1]]
        N = np.cross(verts[tri[1]] - verts[tri[0]], verts[tri[2]] - verts[tri[0]])
        segs.append((ends[0], ends[1]) if np.dot(np.cross(q - p, n), N) >= 0 else (ends[1], ends[0]))
    return segs, points


def _chain(segs, xy, directed=True):
    """Join segments between named nodes into loops.

    directed  follow each segment tail to head. On a closed, consistently wound
              mesh every node has as many segments leaving as arriving, so every
              walk closes. Where bodies touch (a node shared by four faces) the
              walk takes the sharpest left turn, keeping the material it is
              tracing on its left, so touching bodies come out as separate loops.
    undirected  for meshes whose winding is not consistent: segments are joined
              whichever way round they come.
    Returns (loops as node lists, number of chains that could not close)."""
    if directed:
        out, balance = {}, {}
        for a, b in segs:
            out.setdefault(a, []).append(b)
            balance[a] = balance.get(a, 0) + 1
            balance[b] = balance.get(b, 0) - 1
        def turn(prev, cur, nxt):
            (x0, y0), (x1, y1), (x2, y2) = xy[prev], xy[cur], xy[nxt]
            ax, ay, bx, by = x1 - x0, y1 - y0, x2 - x1, y2 - y1
            return math.atan2(ax * by - ay * bx, ax * bx + ay * by)

        loops = []
        for start in list(out):
            while out.get(start):
                loop, prev, cur = [start], start, out[start].pop()
                while cur != start and out.get(cur):
                    loop.append(cur)
                    options = out[cur]
                    i = max(range(len(options)), key=lambda j: turn(prev, cur, options[j])) \
                        if len(options) > 1 else 0
                    prev, cur = cur, options.pop(i)
                if cur == start and len(loop) >= 3:
                    loops.append(loop)
        return loops, sum(v for v in balance.values() if v > 0)
    adj = {}
    for k, (a, b) in enumerate(segs):
        adj.setdefault(a, []).append(k)
        adj.setdefault(b, []).append(k)
    used = [False] * len(segs)
    loops, open_count = [], 0
    # chain ends first, so an open chain is walked whole
    order = sorted(adj, key=lambda node: len(adj[node]) % 2 == 0)
    for start in order:
        while True:
            free = [k for k in adj[start] if not used[k]]
            if not free:
                break
            loop, cur, k = [start], start, free[0]
            while True:
                used[k] = True
                a, b = segs[k]
                cur = b if a == cur else a
                if cur == start:
                    break
                loop.append(cur)
                nxt = [j for j in adj[cur] if not used[j]]
                if not nxt:
                    break
                k = nxt[0]
            if cur == start and len(loop) >= 3:
                loops.append(loop)
            elif cur != start:
                open_count += 1
    return loops, open_count


def _area(loop):
    p = np.asarray(loop)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _inside(pts, loop):
    """Even-odd point-in-polygon for an (m, 2) array of points."""
    pts = np.atleast_2d(np.asarray(pts, float))
    p = np.asarray(loop, float)
    xi, yi = p[:, 0], p[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    x, y = pts[:, :1], pts[:, 1:2]
    cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
    return np.count_nonzero(cross, axis=1) % 2 == 1


def _within(loop, other, samples=64):
    """True if most of the loop's vertices and edge midpoints lie inside 'other'.
    A single test point can sit exactly on a loop that touches this one at a
    point or along an edge; a majority of points cannot all do so."""
    p = np.asarray(loop, float)
    pts = np.vstack([p, (p + np.roll(p, -1, axis=0)) / 2])
    if len(pts) > samples:
        pts = pts[np.linspace(0, len(pts) - 1, samples).astype(int)]
    return 2 * int(np.count_nonzero(_inside(pts, other))) > len(pts)


def nest(loops):
    """Group loops of either winding into (outer, holes): a loop inside an even
    number of others is an outer boundary, inside an odd number a hole of its
    immediate parent."""
    loops = sorted(loops, key=lambda l: -abs(_area(l)))
    depth, parent = [], []
    for i, l in enumerate(loops):
        containers = [j for j in range(i) if _within(l, loops[j])]
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


def _by_winding(loops):
    """(outer, holes) from loops with material on their left: counter-clockwise
    loops are outer boundaries, clockwise ones holes. Each hole goes to the
    smallest outer it lies within."""
    outers = sorted([l for l in loops if _area(l) > 0], key=_area)
    shapes = [(o, []) for o in outers]
    for h in (l for l in loops if _area(l) < 0):
        for o, holes in shapes:
            if _area(o) > -_area(h) and _within(h, o):
                holes.append(h)
                break
        else:
            warnings.warn('a hole with no outer boundary around it was dropped')
    return shapes


def section(tris, origin, normal, up=None, tol=None):
    """Cut face of the mesh on a plane, as [(outer, holes), ...] in plane (u, v)
    coordinates measured from 'origin'. 'up' picks the plane's v axis.

    Vertices closer than tol (default 1e-9 of the part's size) are welded, then
    the cut is traced through the mesh's own edges, so no distance tolerance
    decides which points join. Warns if a chain of cut segments fails to close,
    which means the mesh is not watertight there."""
    tris = np.asarray(tris, float)
    if len(tris) == 0:
        raise ValueError('empty mesh')
    origin = np.asarray(origin, float)
    u, v, n = plane_basis(normal, up)
    size = float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0))) or 1.0
    verts, faces = _weld(tris, tol or 1e-9 * size)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if np.einsum('ij,ij->i', a, np.cross(b, c)).sum() < 0:     # wound inward: flip it
        faces = faces[:, ::-1]
    segs, points = _cut(verts, faces, origin, n, 1e-12 * size)
    flat = {k: (float(np.dot(p - origin, u)), float(np.dot(p - origin, v))) for k, p in points.items()}
    loops, open_count = _chain(segs, flat)
    if open_count:
        # a closed mesh with faces wound against their neighbours still cuts into
        # closed loops if direction is ignored; nest those by containment instead
        loops2, open2 = _chain(segs, flat, directed=False)
        if open2 < open_count:
            warnings.warn('the mesh winding is inconsistent; loops were joined ignoring direction')
            loops, open_count = loops2, open2
            if open_count:
                warnings.warn('%d open chain(s) in the section: the mesh is not closed there' % open_count)
            return nest([[flat[k] for k in l] for l in loops])
        warnings.warn('%d open chain(s) in the section: the mesh is not closed there' % open_count)
    return _by_winding([[flat[k] for k in l] for l in loops])


def area(shapes):
    """Net area of [(outer, holes)] shapes."""
    return sum(abs(_area(o)) - sum(abs(_area(h)) for h in holes) for o, holes in shapes)
