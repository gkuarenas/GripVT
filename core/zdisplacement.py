H_MM: float = 24.6
POISSON: float = 0.495 # common Poisson's ratio approximation for elastomers 
T_MM: float = 4.1

_A: float = (H_MM * POISSON + T_MM) / (T_MM * H_MM)
A_INV: float = 1.0 / _A


def compute(
    dx_px: float,
    dy_px: float,
    area_baseline: float,
    area_current: float,
    fx: float,
    fy: float,
) -> tuple[float, float, float]:
    dx_mm = dx_px * (H_MM / fx)
    dy_mm = dy_px * (H_MM / fy)

    if area_baseline <= 0.0:
        delta_z_mm = 0.0
    else:
        alpha = (area_current - area_baseline) / area_baseline
        delta_z_mm = max(0.0, alpha * A_INV)

    return dx_mm, dy_mm, delta_z_mm
