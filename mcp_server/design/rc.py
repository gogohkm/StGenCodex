# mcp_server/design/rc.py
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Optional

from .units import N_to_kN, Nmm_to_kNm


@dataclass
class RcBeamRectInputs:
    # Geometry (mm)
    b: float
    h: Optional[float] = None
    d: Optional[float] = None
    cover: Optional[float] = None
    stirrup_dia: Optional[float] = None
    bar_dia: Optional[float] = None

    # Materials (MPa)
    fc: float = 24.0
    fy: float = 400.0

    # Reinforcement
    As: Optional[float] = None
    As_top: Optional[float] = None
    As_bot: Optional[float] = None
    Av: Optional[float] = None
    s: Optional[float] = None

    # Strength reduction
    phi_flex: float = 0.9
    phi_shear: float = 0.75


@dataclass
class RcColumnAxialInputs:
    # Geometry (mm)
    Ag: float
    As: float

    # Materials (MPa)
    fc: float = 24.0
    fy: float = 400.0

    phi_axial: float = 0.65


def _effective_depth(d: Optional[float], h: Optional[float], cover: Optional[float],
                     stirrup_dia: Optional[float], bar_dia: Optional[float]) -> Optional[float]:
    if d is not None:
        return float(d)
    if h is None:
        return None
    if cover is None or stirrup_dia is None or bar_dia is None:
        return 0.9 * float(h)
    return float(h) - float(cover) - float(stirrup_dia) - 0.5 * float(bar_dia)


def _mn_from_As(As: float, b: float, d: float, fc: float, fy: float) -> float:
    a = (As * fy) / (0.85 * fc * b)
    Mn = As * fy * (d - 0.5 * a)
    return Mn


def rc_beam_rect_capacity(inputs: RcBeamRectInputs) -> Dict[str, Any]:
    d_eff = _effective_depth(inputs.d, inputs.h, inputs.cover, inputs.stirrup_dia, inputs.bar_dia)
    if d_eff is None:
        raise ValueError("effective depth could not be determined")

    b = float(inputs.b)
    fc = float(inputs.fc)
    fy = float(inputs.fy)

    trace: Dict[str, Any] = {"d_eff": d_eff}

    Mn_pos = None
    Mn_neg = None
    Mn = None

    if inputs.As_bot:
        Mn_pos = inputs.phi_flex * _mn_from_As(float(inputs.As_bot), b, d_eff, fc, fy)
        trace["As_bot"] = float(inputs.As_bot)
    if inputs.As_top:
        Mn_neg = inputs.phi_flex * _mn_from_As(float(inputs.As_top), b, d_eff, fc, fy)
        trace["As_top"] = float(inputs.As_top)

    if inputs.As and Mn_pos is None and Mn_neg is None:
        Mn = inputs.phi_flex * _mn_from_As(float(inputs.As), b, d_eff, fc, fy)
        trace["As"] = float(inputs.As)
    else:
        if Mn_pos is not None and Mn_neg is not None:
            Mn = min(Mn_pos, Mn_neg)
        elif Mn_pos is not None:
            Mn = Mn_pos
        elif Mn_neg is not None:
            Mn = Mn_neg

    Vn = None
    if inputs.Av and inputs.s:
        Av = float(inputs.Av)
        s = float(inputs.s)
        Vc = 0.17 * sqrt(fc) * b * d_eff
        Vs = Av * fy * d_eff / s
        Vn = inputs.phi_shear * (Vc + Vs)
        trace["Av"] = Av
        trace["s"] = s
    else:
        Vc = 0.17 * sqrt(fc) * b * d_eff
        Vn = inputs.phi_shear * Vc

    out = {
        "Mn": Nmm_to_kNm(Mn) if Mn is not None else None,
        "Mn_pos": Nmm_to_kNm(Mn_pos) if Mn_pos is not None else None,
        "Mn_neg": Nmm_to_kNm(Mn_neg) if Mn_neg is not None else None,
        "Vn": N_to_kN(Vn) if Vn is not None else None,
        "trace": trace,
    }
    return out


def rc_column_axial_capacity(inputs: RcColumnAxialInputs) -> Dict[str, Any]:
    Ag = float(inputs.Ag)
    As = float(inputs.As)
    fc = float(inputs.fc)
    fy = float(inputs.fy)

    Pn = inputs.phi_axial * (0.85 * fc * (Ag - As) + fy * As)
    return {
        "Pn": N_to_kN(Pn),
        "trace": {"Ag": Ag, "As": As, "fc": fc, "fy": fy},
    }
