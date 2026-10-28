from .build_local_dcbf import build_local_dcbf


def wrapper_cost(U, params, field):
    return build_local_dcbf(U, params, field, return_details=False)
