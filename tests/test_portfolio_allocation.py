import numpy as np

from tema.portfolio import allocate_portfolio_weights


def test_allocate_portfolio_weights_bl_shape_sum_and_bounds():
    returns_window = np.array(
        [
            [0.01, 0.00, -0.01],
            [0.02, -0.01, 0.00],
            [0.00, 0.01, 0.02],
            [-0.01, 0.00, 0.01],
        ],
        dtype=float,
    )
    result = allocate_portfolio_weights(
        expected_alphas=[0.02, 0.01, -0.01],
        returns_window=returns_window,
        signals=[1.0, 0.5, -0.2],
        method="bl",
        min_weight=0.0,
        max_weight=1.0,
    )

    assert len(result.weights) == 3
    assert abs(sum(result.weights) - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in result.weights)


def test_allocate_portfolio_weights_hooks_are_deterministic_fallbacks():
    result_hrp = allocate_portfolio_weights(
        expected_alphas=[0.01, -0.01, 0.03],
        returns_window=None,
        method="hrp",
    )
    result_nco = allocate_portfolio_weights(
        expected_alphas=[0.01, -0.01, 0.03],
        returns_window=None,
        method="nco",
    )

    assert result_hrp.used_fallback is True
    assert result_nco.used_fallback is True
    assert abs(sum(result_hrp.weights) - 1.0) < 1e-9
    assert abs(sum(result_nco.weights) - 1.0) < 1e-9
