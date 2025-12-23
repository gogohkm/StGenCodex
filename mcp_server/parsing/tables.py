# mcp_server/parsing/tables.py
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TextPoint:
    cad_entity_id: int
    x: float
    y: float
    text: str
    layer: Optional[str] = None


def _auto_tol(vals: List[float], fallback: float) -> float:
    if len(vals) < 10:
        return fallback
    v = sorted(vals)
    diffs = []
    for i in range(1, len(v)):
        d = abs(v[i] - v[i - 1])
        if d > 1e-6:
            diffs.append(d)
    if not diffs:
        return fallback
    m = median(diffs)
    return max(2.0, min(25.0, float(m) * 0.35))


def _cluster_1d(values: List[float], tol: float) -> List[float]:
    if not values:
        return []
    vals = sorted(values)
    centers = [vals[0]]
    counts = [1]
    for v in vals[1:]:
        if abs(v - centers[-1]) <= tol:
            c = centers[-1] * counts[-1] + v
            counts[-1] += 1
            centers[-1] = c / counts[-1]
        else:
            centers.append(v)
            counts.append(1)
    return centers


def extract_grid_tables(
    points: List[TextPoint],
    min_rows: int = 3,
    min_cols: int = 2,
    min_cells: int = 12,
    row_tol: Optional[float] = None,
    col_tol: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Simple grid table inference from clustered x/y coordinates.
    """
    if len(points) < min_cells:
        return []

    ys = [p.y for p in points]
    xs = [p.x for p in points]
    rt = row_tol if row_tol is not None else _auto_tol(ys, fallback=6.0)
    ct = col_tol if col_tol is not None else _auto_tol(xs, fallback=8.0)

    row_centers = _cluster_1d(ys, rt)
    if len(row_centers) < min_rows:
        return []

    col_centers = _cluster_1d(xs, ct)
    if len(col_centers) < min_cols:
        return []

    cells: Dict[Tuple[int, int], List[TextPoint]] = {}
    for p in points:
        rbest = None
        rdist = None
        for ri, yc in enumerate(row_centers):
            d = abs(p.y - yc)
            if rdist is None or d < rdist:
                rdist = d
                rbest = ri
        cbest = None
        cdist = None
        for ci, xc in enumerate(col_centers):
            d = abs(p.x - xc)
            if cdist is None or d < cdist:
                cdist = d
                cbest = ci
        if rdist is None or cdist is None:
            continue
        if rdist > rt or cdist > ct:
            continue
        cells.setdefault((rbest, cbest), []).append(p)

    rows = len(row_centers)
    cols = len(col_centers)
    filled = len(cells)
    occupancy = filled / max(1, rows * cols)

    if rows < min_rows or cols < min_cols or filled < min_cells:
        return []

    minx = min(xs)
    maxx = max(xs)
    miny = min(ys)
    maxy = max(ys)

    table_cells = []
    for (ri, ci), ps in cells.items():
        ps_sorted = sorted(ps, key=lambda p: p.x)
        text = " ".join([p.text.strip() for p in ps_sorted if p.text.strip()])
        rep = ps_sorted[0]
        table_cells.append({
            "row_idx": int(ri),
            "col_idx": int(ci),
            "cad_entity_id": int(rep.cad_entity_id),
            "text": text,
            "x": float(rep.x),
            "y": float(rep.y),
        })

    conf = 0.4 + 0.6 * min(1.0, occupancy)

    return [{
        "method": "grid",
        "rows": rows,
        "cols": cols,
        "confidence": conf,
        "bbox": {"minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy)},
        "meta": {"row_tol": rt, "col_tol": ct, "occupancy": occupancy},
        "cells": table_cells,
    }]
