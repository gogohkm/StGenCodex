# mcp_server/design/steel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .units import N_to_kN, Nmm_to_kNm


@dataclass
class SteelBeamInputs:
    Fy: float = 345.0
    Zx: Optional[float] = None
    Aw: Optional[float] = None
    phi_flex: float = 0.9
    phi_shear: float = 0.9


def steel_beam_capacity(inputs: SteelBeamInputs) -> Dict[str, Any]:
    Fy = float(inputs.Fy)
    Mn = None
    Vn = None

    if inputs.Zx is not None:
        Mn = inputs.phi_flex * Fy * float(inputs.Zx)
    if inputs.Aw is not None:
        Vn = inputs.phi_shear * 0.6 * Fy * float(inputs.Aw)

    return {
        "Mn": Nmm_to_kNm(Mn) if Mn is not None else None,
        "Vn": N_to_kN(Vn) if Vn is not None else None,
        "trace": {"Fy": Fy, "Zx": inputs.Zx, "Aw": inputs.Aw},
    }
