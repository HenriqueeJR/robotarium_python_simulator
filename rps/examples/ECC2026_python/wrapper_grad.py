from .build_local_dcbf import build_local_dcbf


def wrapper_grad(U, params, field):
    return build_local_dcbf(U, params, field)[1]
