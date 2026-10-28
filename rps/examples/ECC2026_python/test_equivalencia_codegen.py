from __future__ import annotations

import numpy as np

from .build_local_dcbf import build_local_dcbf, _build_local_dcbf_python
from .make_local_params import make_local_params
from .pack_params_1D import pack_params_1D, unpack_neighbors
from .wrapper_cost import wrapper_cost
from .wrapper_grad import wrapper_grad


def _test_inputs():
    horizon = 8
    corridors = [{"v": np.array([
        [-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [-1.0, 0.5]
    ])}]
    roundabouts = np.array([[0.5, 0.5, 0.35, 0.08, 1.0]])
    x_vec = np.linspace(-1.0, 1.0, 21)
    y_vec = np.linspace(-1.0, 1.0, 21)
    X, Y = np.meshgrid(x_vec, y_vec)
    field = {
        "x_vec": x_vec, "y_vec": y_vec,
        "ux": 0.06 + 0.01 * X - 0.005 * Y,
        "uy": 0.01 + 0.004 * X + 0.008 * Y,
    }
    x0 = np.array([-0.6, 0.0, 0.0])
    target = np.array([0.8, 0.4])
    previous = np.array([0.04, 0.0])
    controls = np.vstack((
        0.075 + 0.005 * np.sin(np.arange(1, horizon + 1)),
        0.04 * np.cos(np.arange(1, horizon + 1)),
    ))
    U = controls.reshape(-1, order="F")
    neighbors = np.zeros((2, horizon + 1, 3))
    for neighbor in range(3):
        neighbors[:, :, neighbor] = np.array([
            np.linspace(0.2 + neighbor * 0.1, 0.5 + neighbor * 0.1, horizon + 1),
            np.full(horizon + 1, 0.8 - neighbor * 0.1),
        ])
    return horizon, corridors, roundabouts, field, x0, target, previous, U, neighbors


def test_parameter_pack_round_trip_and_wrappers():
    (horizon, corridors, roundabouts, field, x0, target, previous,
     U, neighbors) = _test_inputs()
    packed = pack_params_1D(
        x0, neighbors, previous, target, horizon, 0.033, 0.27, 0.1,
        2e8, 0.05, corridors, roundabouts, 1,
    )
    decoded = unpack_neighbors(packed, horizon, 3)
    assert np.array_equal(decoded, neighbors)

    params = make_local_params(
        x0, neighbors, previous, target, horizon, 0.033, 0.27, 0.1,
        2e8, 0.05, corridors, roundabouts,
    )
    objective = build_local_dcbf(U, params, field)
    assert np.isclose(wrapper_cost(U, params, field), objective[0])
    assert np.array_equal(wrapper_grad(U, params, field), objective[1])

    reference = _build_local_dcbf_python(U, params, field)
    assert np.isclose(objective[0], reference[0], rtol=1e-12, atol=1e-12)
    assert np.allclose(objective[1], reference[1], rtol=1e-10, atol=1e-10)
    assert np.allclose(objective[2], reference[2], rtol=1e-12, atol=1e-12)
    for key in objective[3]:
        assert np.isclose(objective[3][key], reference[3][key], rtol=1e-12, atol=1e-12)


def test_cost_gradient_are_finite():
    (horizon, corridors, roundabouts, field, x0, target, previous,
     U, neighbors) = _test_inputs()
    params = make_local_params(
        x0, neighbors, previous, target, horizon, 0.033, 0.27, 0.1,
        2e8, 0.05, corridors, roundabouts,
    )
    cost, gradient, _, _ = build_local_dcbf(U, params, field)
    assert np.isfinite(cost)
    assert np.all(np.isfinite(gradient))
