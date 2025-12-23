# mcp_server/design/units.py

def N_to_kN(x: float) -> float:
    return x / 1e3


def Nmm_to_kNm(x: float) -> float:
    # 1 kN-m = 1e6 N-mm
    return x / 1e6


def MPa_is_N_per_mm2(x: float) -> float:
    # 1 MPa == 1 N/mm^2
    return x
