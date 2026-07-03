import numpy as np
import pytest

from core.blobfit import estimate_background, fit_gaussian_logparabola, fit_moments


def _make_gaussian_window(
    size: int,
    amplitude: float,
    sigma_x: float,
    sigma_y: float,
    x0: float,
    y0: float,
    background: float,
    clip_at: float | None = None,
) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    xs = xs.astype(float)
    ys = ys.astype(float)
    values = background + amplitude * np.exp(
        -(((xs - x0) ** 2) / (2 * sigma_x**2) + ((ys - y0) ** 2) / (2 * sigma_y**2))
    )
    if clip_at is not None:
        values = np.minimum(values, clip_at)
    return values


def test_worked_example_noiseless_isotropic() -> None:
    window = _make_gaussian_window(
        size=9, amplitude=100.0, sigma_x=1.2, sigma_y=1.2, x0=4.0, y0=4.0, background=0.0
    )
    fit = fit_gaussian_logparabola(window, background=0.0)
    assert fit is not None
    assert fit.x0 == pytest.approx(4.0, abs=0.01)
    assert fit.y0 == pytest.approx(4.0, abs=0.01)
    assert fit.sigma_x == pytest.approx(1.2, abs=0.01)
    assert fit.sigma_y == pytest.approx(1.2, abs=0.01)
    assert fit.amplitude == pytest.approx(100.0, abs=1.0)
    assert fit.residual < 1e-6


def test_elliptical_offcenter() -> None:
    window = _make_gaussian_window(
        size=11, amplitude=150.0, sigma_x=1.0, sigma_y=2.0, x0=5.3, y0=4.7, background=5.0
    )
    fit = fit_gaussian_logparabola(window, background=5.0)
    assert fit is not None
    assert fit.x0 == pytest.approx(5.3, abs=0.02)
    assert fit.y0 == pytest.approx(4.7, abs=0.02)
    assert fit.sigma_x == pytest.approx(1.0, abs=0.02)
    assert fit.sigma_y == pytest.approx(2.0, abs=0.02)
    assert fit.amplitude == pytest.approx(150.0, rel=0.02)


def test_noisy_gaussian_recovers_within_tolerance() -> None:
    clean = _make_gaussian_window(
        size=13, amplitude=200.0, sigma_x=1.5, sigma_y=1.5, x0=6.0, y0=6.0, background=20.0
    )
    rng = np.random.default_rng(42)
    noisy = clean + rng.normal(0.0, 3.0, size=clean.shape)

    fit = fit_gaussian_logparabola(noisy, background=20.0)
    assert fit is not None
    assert fit.x0 == pytest.approx(6.0, abs=0.3)
    assert fit.y0 == pytest.approx(6.0, abs=0.3)
    assert fit.sigma_x == pytest.approx(1.5, rel=0.2)
    assert fit.sigma_y == pytest.approx(1.5, rel=0.2)


def test_invalid_no_peak_returns_none() -> None:
    ys, xs = np.mgrid[0:9, 0:9]
    xs = xs.astype(float)
    ys = ys.astype(float)
    # Brightness increases AWAY from center -- an inverted peak, no maximum
    # for the log-parabola to lock onto.
    window = 50.0 + 0.5 * ((xs - 4.0) ** 2 + (ys - 4.0) ** 2)
    fit = fit_gaussian_logparabola(window, background=0.0)
    assert fit is None


def test_invalid_flat_window_returns_none() -> None:
    window = np.full((9, 9), 50.0)
    fit = fit_gaussian_logparabola(window, background=50.0)
    assert fit is None


def test_invalid_window_too_small_returns_none() -> None:
    window = np.full((2, 2), 100.0)
    fit = fit_gaussian_logparabola(window, background=0.0)
    assert fit is None


def test_saturated_core_masked_recovers_from_wings() -> None:
    window = _make_gaussian_window(
        size=15, amplitude=300.0, sigma_x=2.5, sigma_y=2.5,
        x0=7.0, y0=7.0, background=10.0, clip_at=254.0,
    )
    fit = fit_gaussian_logparabola(window, background=10.0)
    assert fit is not None
    assert fit.x0 == pytest.approx(7.0, abs=0.3)
    assert fit.y0 == pytest.approx(7.0, abs=0.3)
    assert fit.sigma_x == pytest.approx(2.5, rel=0.15)
    assert fit.sigma_y == pytest.approx(2.5, rel=0.15)


def test_neighbor_mask_prevents_bleed() -> None:
    main = _make_gaussian_window(
        size=15, amplitude=100.0, sigma_x=1.2, sigma_y=1.2, x0=3.0, y0=3.0, background=0.0
    )
    neighbor = _make_gaussian_window(
        size=15, amplitude=90.0, sigma_x=1.2, sigma_y=1.2, x0=11.0, y0=11.0, background=0.0
    )
    combined = main + neighbor

    unmasked = fit_gaussian_logparabola(combined, background=0.0)
    assert unmasked is None or abs(unmasked.x0 - 3.0) > 1.0

    exclude_neighbor = np.ones((15, 15), dtype=bool)
    exclude_neighbor[7:, 7:] = False
    masked = fit_gaussian_logparabola(combined, background=0.0, mask=exclude_neighbor)
    assert masked is not None
    assert masked.x0 == pytest.approx(3.0, abs=0.1)
    assert masked.y0 == pytest.approx(3.0, abs=0.1)


def test_moments_size_matches_gaussian_area_for_true_gaussian() -> None:
    window = _make_gaussian_window(
        size=15, amplitude=100.0, sigma_x=1.3, sigma_y=1.3, x0=7.0, y0=7.0, background=0.0
    )
    moments = fit_moments(window, background=0.0)
    assert moments is not None
    true_size = 2.0 * np.pi * 1.3 * 1.3
    assert moments.size == pytest.approx(true_size, rel=0.05)
    assert moments.x0 == pytest.approx(7.0, abs=0.05)
    assert moments.y0 == pytest.approx(7.0, abs=0.05)


def test_moments_returns_none_on_flat_window() -> None:
    window = np.full((9, 9), 50.0)
    moments = fit_moments(window, background=50.0)
    assert moments is None


def test_estimate_background_uses_border_ring() -> None:
    window = np.full((9, 9), 20.0)
    window[3:6, 3:6] = 200.0
    bg = estimate_background(window, border=2)
    assert bg == pytest.approx(20.0)
