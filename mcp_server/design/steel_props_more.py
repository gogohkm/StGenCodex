# mcp_server/design/steel_props_more.py
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Dict


@dataclass
class SteelBoxDims:
    H: float
    B: float
    t: float


def compute_box_props(d: SteelBoxDims) -> Dict[str, Any]:
    H, B, t = float(d.H), float(d.B), float(d.t)
    if H <= 0 or B <= 0 or t <= 0:
        raise ValueError("H,B,t must be positive.")
    Hi = H - 2 * t
    Bi = B - 2 * t
    if Hi <= 0 or Bi <= 0:
        raise ValueError("Invalid box dims: H-2t and B-2t must be positive.")

    A = B * H - Bi * Hi
    Ix = (B * H ** 3 - Bi * Hi ** 3) / 12.0
    Iy = (H * B ** 3 - Hi * Bi ** 3) / 12.0
    Sx = Ix / (H / 2.0)
    Sy = Iy / (B / 2.0)

    Zx = (B * H ** 2 - Bi * Hi ** 2) / 4.0
    Zy = (H * B ** 2 - Hi * Bi ** 2) / 4.0

    Avx = 2.0 * t * Hi
    Avy = 2.0 * t * Bi

    return {
        "family": "steel_box",
        "dims": {"H_mm": H, "B_mm": B, "t_mm": t},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_mm3": Sx,
            "Sy_mm3": Sy,
            "Zx_mm3": Zx,
            "Zy_mm3": Zy,
            "Aw_mm2": Avx,
            "Avx_mm2": Avx,
            "Avy_mm2": Avy,
        },
    }


@dataclass
class SteelPipeDims:
    D: float
    t: float


def compute_pipe_props(d: SteelPipeDims) -> Dict[str, Any]:
    D, t = float(d.D), float(d.t)
    if D <= 0 or t <= 0:
        raise ValueError("D,t must be positive.")
    Di = D - 2 * t
    if Di <= 0:
        raise ValueError("Invalid pipe dims: D-2t must be positive.")

    Ro = D / 2.0
    Ri = Di / 2.0

    A = (pi / 4.0) * (D ** 2 - Di ** 2)
    Ix = (pi / 64.0) * (D ** 4 - Di ** 4)
    Iy = Ix
    Sx = Ix / (D / 2.0)
    Sy = Sx

    Zx = (4.0 / 3.0) * (Ro ** 3 - Ri ** 3)
    Zy = Zx

    Aw = A

    return {
        "family": "steel_pipe",
        "dims": {"D_mm": D, "t_mm": t},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_mm3": Sx,
            "Sy_mm3": Sy,
            "Zx_mm3": Zx,
            "Zy_mm3": Zy,
            "Aw_mm2": Aw,
        },
    }


@dataclass
class SteelChannelDims:
    H: float
    B: float
    tw: float
    tf: float


def compute_channel_props(d: SteelChannelDims) -> Dict[str, Any]:
    """
    Channel modeled as 3 rectangles (no overlap):
      - web: tw x (H-2tf)
      - two flanges: B x tf (top & bottom)
    Coordinate:
      x from web back face (0..B), y from bottom (0..H)
    """
    H, B, tw, tf = float(d.H), float(d.B), float(d.tw), float(d.tf)
    if H <= 0 or B <= 0 or tw <= 0 or tf <= 0:
        raise ValueError("H,B,tw,tf must be positive.")
    hw = H - 2.0 * tf
    if hw <= 0:
        raise ValueError("Invalid dims: H-2tf must be positive.")
    if tw >= B:
        raise ValueError("Invalid dims: tw must be < B (typical).")

    Aweb = tw * hw
    Af = B * tf
    A = Aweb + 2.0 * Af

    x_web = tw / 2.0
    y_web = H / 2.0
    x_f = B / 2.0
    y_top = H - tf / 2.0

    xbar = (Aweb * x_web + Af * x_f + Af * x_f) / A
    ybar = H / 2.0

    Ix_web = (tw * hw ** 3) / 12.0
    y_f = abs(y_top - ybar)
    Ix_f_each = (B * tf ** 3) / 12.0 + Af * (y_f ** 2)
    Ix = Ix_web + 2.0 * Ix_f_each

    Iy_web = (hw * tw ** 3) / 12.0 + Aweb * ((x_web - xbar) ** 2)
    Iy_f_each = (tf * B ** 3) / 12.0 + Af * ((x_f - xbar) ** 2)
    Iy = Iy_web + 2.0 * Iy_f_each

    Sx = Ix / (H / 2.0)
    c_left = xbar
    c_right = B - xbar
    Sy_left = Iy / c_left if c_left > 0 else None
    Sy_right = Iy / c_right if c_right > 0 else None
    Sy_min = min([v for v in (Sy_left, Sy_right) if v is not None])

    Aweb_top = tw * (hw / 2.0)
    y_w = hw / 4.0
    y_f_pl = (H / 2.0 - tf / 2.0)
    Zx = 2.0 * (Af * y_f_pl + Aweb_top * y_w)

    Aw = tw * hw

    return {
        "family": "steel_channel",
        "dims": {"H_mm": H, "B_mm": B, "tw_mm": tw, "tf_mm": tf},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_mm3": Sx,
            "Sy_min_mm3": Sy_min,
            "Zx_mm3": Zx,
            "Aw_mm2": Aw,
            "centroid_x_mm": xbar,
            "centroid_y_mm": ybar,
        },
        "warnings": [
            "Channel is unsymmetric about y-axis; use catalog values if available for Sy/Zy and design checks about minor/principal axes."
        ],
    }


@dataclass
class SteelAngleDims:
    b: float
    d: float
    t: float


def compute_angle_props(d: SteelAngleDims) -> Dict[str, Any]:
    """
    L-angle as union of two rectangles minus overlap square.
    No plastic modulus exact (PNA not at centroid). Provide conservative Zx= Sx_min if needed elsewhere.
    """
    b, dd, t = float(d.b), float(d.d), float(d.t)
    if b <= 0 or dd <= 0 or t <= 0:
        raise ValueError("b,d,t must be positive.")
    if t >= b or t >= dd:
        raise ValueError("Invalid angle dims: t must be < b and < d.")

    A1 = b * t
    x1, y1 = b / 2.0, t / 2.0
    A2 = t * dd
    x2, y2 = t / 2.0, dd / 2.0
    A3 = t * t
    x3, y3 = t / 2.0, t / 2.0

    A = A1 + A2 - A3
    xbar = (A1 * x1 + A2 * x2 - A3 * x3) / A
    ybar = (A1 * y1 + A2 * y2 - A3 * y3) / A

    def Ix_rect(w, h):
        return w * h ** 3 / 12.0

    def Iy_rect(w, h):
        return h * w ** 3 / 12.0

    Ix = (
        (Ix_rect(b, t) + A1 * (y1 - ybar) ** 2)
        + (Ix_rect(t, dd) + A2 * (y2 - ybar) ** 2)
        - (Ix_rect(t, t) + A3 * (y3 - ybar) ** 2)
    )
    Iy = (
        (Iy_rect(b, t) + A1 * (x1 - xbar) ** 2)
        + (Iy_rect(t, dd) + A2 * (x2 - xbar) ** 2)
        - (Iy_rect(t, t) + A3 * (x3 - xbar) ** 2)
    )

    c_top = dd - ybar
    c_bot = ybar
    Sx_top = Ix / c_top if c_top > 0 else None
    Sx_bot = Ix / c_bot if c_bot > 0 else None
    Sx_min = min([v for v in (Sx_top, Sx_bot) if v is not None])

    c_right = b - xbar
    c_left = xbar
    Sy_right = Iy / c_right if c_right > 0 else None
    Sy_left = Iy / c_left if c_left > 0 else None
    Sy_min = min([v for v in (Sy_right, Sy_left) if v is not None])

    Zx_cons = Sx_min

    return {
        "family": "steel_angle",
        "dims": {"b_mm": b, "d_mm": dd, "t_mm": t},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_min_mm3": Sx_min,
            "Sy_min_mm3": Sy_min,
            "Zx_mm3": Zx_cons,
            "Aw_mm2": A,
            "centroid_x_mm": xbar,
            "centroid_y_mm": ybar,
        },
        "warnings": [
            "Angle plastic modulus is not computed exactly; Zx is set conservatively to Sx_min. Prefer catalog values for design."
        ],
    }
