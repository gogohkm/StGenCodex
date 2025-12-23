# mcp_server/design/steel_props.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def norm_section_name(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "").replace("-", "").replace("×", "X")


@dataclass
class SteelHSectionDims:
    H: float
    B: float
    tw: float
    tf: float


def compute_h_section_props(d: SteelHSectionDims) -> Dict[str, Any]:
    """
    Symmetric I/H section, sharp corners, no fillets.
    Returns: A (mm2), Ix/Iy (mm4), Sx/Sy (mm3), Zx/Zy (mm3), Aw (mm2)
    """
    H, B, tw, tf = float(d.H), float(d.B), float(d.tw), float(d.tf)
    if H <= 0 or B <= 0 or tw <= 0 or tf <= 0:
        raise ValueError("H,B,tw,tf must be positive.")
    if 2 * tf >= H:
        raise ValueError("Invalid dims: 2*tf must be < H.")
    if tw >= B:
        raise ValueError("Invalid dims: tw must be < B for typical H-section.")

    hw = H - 2.0 * tf

    Af = B * tf
    Aw = tw * hw
    A = 2.0 * Af + Aw

    y = H / 2.0 - tf / 2.0
    Ix_flange = (B * tf ** 3) / 12.0 + Af * (y ** 2)
    Ix_web = (tw * hw ** 3) / 12.0
    Ix = 2.0 * Ix_flange + Ix_web

    Iy_flange = (tf * B ** 3) / 12.0
    Iy_web = (hw * tw ** 3) / 12.0
    Iy = 2.0 * Iy_flange + Iy_web

    Sx = Ix / (H / 2.0)
    Sy = Iy / (B / 2.0)

    Zx = 2.0 * Af * (H / 2.0 - tf / 2.0) + Aw * (hw / 4.0)
    Zy = 2.0 * (tf * B ** 2 / 4.0) + (hw * tw ** 2 / 4.0)

    return {
        "A": A,
        "Aw": Aw,
        "Ix": Ix,
        "Iy": Iy,
        "Sx": Sx,
        "Sy": Sy,
        "Zx": Zx,
        "Zy": Zy,
    }
