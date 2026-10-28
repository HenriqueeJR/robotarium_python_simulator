from .build_local_dcbf import build_local_dcbf


def build_local_dcbf_mexable(U, params, field, return_details=True):
    """Python fallback with the same objective; MATLAB Coder MEX is not used."""
    return build_local_dcbf(U, params, field, return_details=return_details)
