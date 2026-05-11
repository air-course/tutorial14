import numpy as np
import sympy as smp

def check_type(x):
    """
    checks the type of x and returns the suitable library
    (pydrake.symbolic, sympy or numpy) for furhter calculations on x.
    """
    if isinstance(x, (tuple, np.ndarray)) and isinstance(x[0], smp.Expr):
        md = smp
    elif isinstance(x, np.ndarray) and x.dtype == object and pydrake_available:
        md = sym
    else:
        md = np
    return md