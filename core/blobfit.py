from dataclasses import dataclass

import numpy as np

DEFAULT_SATURATION_VALUE = 254.0
DEFAULT_BACKGROUND_BORDER_PX = 2
_EPS = 1e-6
_MIN_VALID_PIXELS = 10
_CENTER_MARGIN_PX = 2.0


@dataclass
class GaussianFit:
    x0: float
    y0: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    residual: float

    @property
    def size(self) -> float:
        return 2.0 * np.pi * self.sigma_x * self.sigma_y


@dataclass
class MomentsFit:
    x0: float
    y0: float
    mu20: float
    mu02: float
    mu11: float
    size: float


def estimate_background(window: np.ndarray, border: int = DEFAULT_BACKGROUND_BORDER_PX) -> float:
    h, w = window.shape
    border = max(1, min(border, h // 2, w // 2)) if h > 2 and w > 2 else 1
    ring = np.zeros(window.shape, dtype=bool)
    ring[:border, :] = True
    ring[-border:, :] = True
    ring[:, :border] = True
    ring[:, -border:] = True
    values = window[ring]
    if values.size == 0:
        return float(np.median(window))
    return float(np.median(values))


def _valid_mask(
    window: np.ndarray,
    background: float,
    saturation_value: float,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    intensity = window.astype(float) - background
    valid = (intensity > _EPS) & (window < saturation_value)
    if mask is not None:
        valid &= mask
    return intensity, valid


def fit_gaussian_logparabola(
    window: np.ndarray,
    background: float,
    *,
    mask: np.ndarray | None = None,
    saturation_value: float = DEFAULT_SATURATION_VALUE,
) -> GaussianFit | None:
    h, w = window.shape
    if h < 3 or w < 3:
        return None

    intensity, valid = _valid_mask(window, background, saturation_value, mask)
    if np.count_nonzero(valid) < _MIN_VALID_PIXELS:
        return None

    ys, xs = np.mgrid[0:h, 0:w]
    x = xs[valid].astype(float)
    y = ys[valid].astype(float)
    i = intensity[valid]

    ln_i = np.log(i)
    phi = np.stack([np.ones_like(x), x, y, x * x, y * y], axis=1)

    # Fit on the sqrt(weight)-scaled design matrix rather than forming the
    # normal equations (Phi^T W Phi) explicitly -- lstsq is numerically
    # stable at the condition number of Phi itself, while the normal-equation
    # route squares it. w = i^2 so the bright core drives the fit and log-
    # amplified halo noise doesn't drag sigma around frame to frame.
    sqrt_w = i
    phi_w = phi * sqrt_w[:, None]
    target_w = ln_i * sqrt_w

    try:
        params, _residuals, rank, _sv = np.linalg.lstsq(phi_w, target_w, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < phi.shape[1]:
        return None

    a, b, c, d, e = params
    if not (d < -_EPS and e < -_EPS):
        return None

    sigma_x_sq = -1.0 / (2.0 * d)
    sigma_y_sq = -1.0 / (2.0 * e)
    if not (np.isfinite(sigma_x_sq) and np.isfinite(sigma_y_sq)):
        return None
    sigma_x = float(np.sqrt(sigma_x_sq))
    sigma_y = float(np.sqrt(sigma_y_sq))

    x0 = float(-b / (2.0 * d))
    y0 = float(-c / (2.0 * e))
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return None
    if not (-_CENTER_MARGIN_PX <= x0 <= w - 1 + _CENTER_MARGIN_PX):
        return None
    if not (-_CENTER_MARGIN_PX <= y0 <= h - 1 + _CENTER_MARGIN_PX):
        return None

    log_amplitude = a + x0 * b / 2.0 + y0 * c / 2.0
    amplitude = float(np.exp(log_amplitude))
    if not np.isfinite(amplitude):
        return None

    predicted = phi @ params
    residual = float(np.sqrt(np.mean((ln_i - predicted) ** 2)))

    return GaussianFit(
        x0=x0, y0=y0, sigma_x=sigma_x, sigma_y=sigma_y,
        amplitude=amplitude, residual=residual,
    )


def fit_moments(
    window: np.ndarray,
    background: float,
    *,
    mask: np.ndarray | None = None,
    saturation_value: float = DEFAULT_SATURATION_VALUE,
) -> MomentsFit | None:
    h, w = window.shape
    intensity, valid = _valid_mask(window, background, saturation_value, mask)
    if np.count_nonzero(valid) < _MIN_VALID_PIXELS:
        return None

    ys, xs = np.mgrid[0:h, 0:w]
    x = xs[valid].astype(float)
    y = ys[valid].astype(float)
    i = intensity[valid]

    total = float(np.sum(i))
    if total <= _EPS:
        return None

    x0 = float(np.sum(i * x) / total)
    y0 = float(np.sum(i * y) / total)

    dx = x - x0
    dy = y - y0
    mu20 = float(np.sum(i * dx * dx) / total)
    mu02 = float(np.sum(i * dy * dy) / total)
    mu11 = float(np.sum(i * dx * dy) / total)

    det = mu20 * mu02 - mu11 * mu11
    size = 2.0 * np.pi * float(np.sqrt(det)) if det > 0.0 else 0.0

    return MomentsFit(x0=x0, y0=y0, mu20=mu20, mu02=mu02, mu11=mu11, size=size)
