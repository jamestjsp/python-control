# slicot_compat.py - compatibility wrappers for slicot/slycot packages
#
# This module provides wrappers around the slicot package functions to match
# the API used by the older slycot package. It also supports falling back to
# slycot if slicot is not installed.

"""Compatibility layer for slicot/slycot packages.

This module wraps slicot functions to match the slycot API, minimizing changes
to existing code in python-control. If slicot is not installed, it falls back
to using slycot directly.
"""

import numpy as np

from .exception import ControlArgument

# Try to import slicot first (preferred), fall back to slycot
_use_slicot = False
_use_slycot = False

try:
    import slicot  # noqa: F401
    _use_slicot = True
except ImportError:
    try:
        import slycot  # noqa: F401
        _use_slycot = True
    except ImportError:
        pass

if not _use_slicot and not _use_slycot:
    raise ImportError("Neither slicot nor slycot is installed")

__all__ = [
    'SlicotResultWarning', 'SlicotArithmeticError',
    'sb03md', 'sb03od', 'sb04md', 'sb04qd', 'sg03ad',
    'sb02md', 'sb02mt', 'sg02ad', 'sb01bd',
    'sb10ad', 'sb10hd', 'ab08nd',
    'ab09ad', 'ab09md', 'ab09nd',
    'ab13bd', 'ab13dd', 'ab13md',
    'tb01pd', 'tb04ad', 'tb05ad', 'td04ad', 'mb03rd',
]


class SlicotResultWarning(UserWarning):
    """Warning for non-fatal issues from SLICOT routines."""
    pass


class SlicotArithmeticError(ArithmeticError):
    """Error for arithmetic failures in SLICOT routines."""

    def __init__(self, message, info=0):
        super().__init__(message)
        self.info = info


def _check_info(info, routine_name, warn_codes=None):
    """Check info code and raise appropriate exception.

    Parameters
    ----------
    info : int
        Info code returned by SLICOT routine.
    routine_name : str
        Name of the routine for error messages.
    warn_codes : list of int, optional
        Info codes that should generate warnings instead of errors.
    """
    if info == 0:
        return
    if warn_codes and info in warn_codes:
        import warnings
        warnings.warn(
            f"{routine_name} returned info={info}",
            SlicotResultWarning
        )
        return
    if info < 0:
        raise ControlArgument(f"{routine_name}: parameter {-info} is invalid")
    raise SlicotArithmeticError(
        f"{routine_name} returned info={info}", info=info
    )


def sb03md(n, C, A, U, dico, job='X', fact='N', trana='N', ldwork=None):
    """Solve Lyapunov equation (slycot-compatible wrapper).

    slycot API: X, scale, sep, ferr, w = sb03md(n, C, A, U, dico, job, fact, trana)
    slicot API: x, a, u, wr, wi, scale, sep, ferr, info =
                sb03md(dico, job, fact, trana, n, a, c, u)

    Returns
    -------
    X : ndarray
        Solution matrix.
    scale : float
        Scale factor.
    sep : float
        Separation estimate.
    ferr : float
        Forward error bound.
    w : ndarray
        Eigenvalues of A (complex).
    """
    from slicot import sb03md as _sb03md

    A_copy = np.asfortranarray(A.copy())
    C_copy = np.asfortranarray(C.copy())
    U_copy = np.asfortranarray(U.copy()) if fact == 'F' else None

    if fact == 'F':
        X, A_out, U_out, wr, wi, scale, sep, ferr, info = _sb03md(
            dico, job, fact, trana, n, A_copy, C_copy, U_copy
        )
    else:
        X, A_out, U_out, wr, wi, scale, sep, ferr, info = _sb03md(
            dico, job, fact, trana, n, A_copy, C_copy
        )

    _check_info(info, 'sb03md', warn_codes=[1, 2])

    w = wr + 1j * wi

    return X, scale, sep, ferr, w


def sb03od(n, m, A, Q, B, dico, fact='N', trans='N', ldwork=None):
    """Solve Lyapunov equation with Cholesky factor (slycot-compatible wrapper).

    slycot API: X, scale, w = sb03od(n, m, A, Q, B, dico, fact, trans)
    slicot API: u, q_out, wr, wi, scale, info = sb03od(dico, fact, trans, a, b, [q])

    Returns
    -------
    X : ndarray
        Cholesky factor of solution.
    scale : float
        Scale factor.
    w : ndarray
        Eigenvalues of A (complex).
    """
    from slicot import sb03od as _sb03od

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())

    if fact == 'F':
        Q_copy = np.asfortranarray(Q.copy())
        u, q_out, wr, wi, scale, info = _sb03od(
            dico, fact, trans, A_copy, B_copy, Q_copy
        )
    else:
        u, q_out, wr, wi, scale, info = _sb03od(
            dico, fact, trans, A_copy, B_copy
        )

    _check_info(info, 'sb03od')

    w = wr + 1j * wi

    return u, scale, w


def sb04md(n, m, A, B, C, ldwork=None):
    """Solve Sylvester equation AX + XB = C (slycot-compatible wrapper).

    slycot API: X = sb04md(n, m, A, B, C)
    slicot API: x, z, info = sb04md(a, b, c)

    Returns
    -------
    X : ndarray
        Solution matrix.
    """
    from slicot import sb04md as _sb04md

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())

    X, Z, info = _sb04md(A_copy, B_copy, C_copy)

    _check_info(info, 'sb04md')

    return X


def sb04qd(n, m, A, B, C, ldwork=None):
    """Solve discrete Sylvester equation AXB + X = C (slycot-compatible wrapper).

    slycot API: X = sb04qd(n, m, A, B, C)
    slicot API: x, z, info = sb04qd(a, b, c)

    Returns
    -------
    X : ndarray
        Solution matrix.
    """
    from slicot import sb04qd as _sb04qd

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())

    X, Z, info = _sb04qd(A_copy, B_copy, C_copy)

    _check_info(info, 'sb04qd')

    return X


def sg03ad(dico, job, fact, trana, uplo, n, A, E, Q, Z, B, ldwork=None):
    """Solve generalized Lyapunov equation (slycot-compatible wrapper).

    slycot API: A, E, Q, Z, X, scale, sep, ferr, alphar, alphai, beta =
                sg03ad(dico, job, fact, trana, uplo, n, A, E, Q, Z, B)
    slicot API: x, scale, sep, ferr, alphar, alphai, beta, a, e, q, z, info =
                sg03ad(dico, job, fact, trans, uplo, n, a, e, x, q, z)

    Returns
    -------
    A : ndarray
        Updated A matrix (Schur form).
    E : ndarray
        Updated E matrix.
    Q : ndarray
        Orthogonal transformation Q.
    Z : ndarray
        Orthogonal transformation Z.
    X : ndarray
        Solution matrix.
    scale : float
        Scale factor.
    sep : float
        Separation estimate.
    ferr : float
        Forward error bound.
    alphar, alphai, beta : ndarray
        Generalized eigenvalues.
    """
    from slicot import sg03ad as _sg03ad

    A_copy = np.asfortranarray(A.copy())
    E_copy = np.asfortranarray(E.copy())
    B_copy = np.asfortranarray(B.copy())

    if fact == 'F':
        Q_copy = np.asfortranarray(Q.copy())
        Z_copy = np.asfortranarray(Z.copy())
        X, scale, sep, ferr, alphar, alphai, beta, A_out, E_out, Q_out, Z_out, info = _sg03ad(
            dico, job, fact, trana, uplo, n, A_copy, E_copy, B_copy, Q_copy, Z_copy
        )
    else:
        X, scale, sep, ferr, alphar, alphai, beta, A_out, E_out, Q_out, Z_out, info = _sg03ad(
            dico, job, fact, trana, uplo, n, A_copy, E_copy, B_copy
        )

    _check_info(info, 'sg03ad', warn_codes=[1, 2, 3])

    return A_out, E_out, Q_out, Z_out, X, scale, sep, ferr, alphar, alphai, beta


def sb02md(n, A, G, Q, dico, hinv='D', uplo='U', scal='N', sort='S', ldwork=None):
    """Solve algebraic Riccati equation (slycot-compatible wrapper).

    slycot API: X, rcond, w, S, U, A_inv = sb02md(n, A, G, Q, dico, hinv, uplo, scal, sort)
    slicot API: X, rcond, wr, wi, S, U, info = sb02md(dico, hinv, uplo, scal, sort, n, A, G, Q)

    Returns
    -------
    X : ndarray
        Solution matrix.
    rcond : float
        Reciprocal condition number.
    w : ndarray
        Closed-loop eigenvalues (complex).
    S : ndarray
        Schur form.
    U : ndarray
        Orthogonal transformation.
    A_inv : ndarray
        Inverse of A (if computed).
    """
    from slicot import sb02md as _sb02md

    A_copy = np.asfortranarray(A.copy())
    G_copy = np.asfortranarray(G.copy())
    Q_copy = np.asfortranarray(Q.copy())

    X, rcond, wr, wi, S, U, info = _sb02md(
        dico, hinv, uplo, scal, sort, n, A_copy, G_copy, Q_copy
    )

    _check_info(info, 'sb02md')

    w = wr + 1j * wi
    A_inv = A_copy

    return X, rcond, w, S, U, A_inv


def sb02mt(n, m, B, R, A=None, Q=None, L=None, fact='N', jobl='Z', uplo='U', ldwork=None):
    """Prepare data for Riccati solver (slycot-compatible wrapper).

    slycot API: A_b, B_b, Q_b, R_b, L_b, ipiv, oufact, G = sb02mt(n, m, B, R, ...)
    slicot API (jobg='G', jobl='Z'): G, oufact, info = sb02mt(...)

    Returns
    -------
    A_b : ndarray
        Input matrix A (unchanged).
    B_b : ndarray
        Input matrix B (unchanged).
    Q_b : ndarray
        Input matrix Q (unchanged).
    R_b : ndarray
        Factored R matrix.
    L_b : ndarray
        Cross-weighting matrix.
    ipiv : ndarray
        Pivot indices (empty for slicot).
    oufact : int
        Output factorization flag.
    G : ndarray
        G = B * inv(R) * B'.
    """
    from slicot import sb02mt as _sb02mt

    if A is None:
        A = np.zeros((n, n), dtype=float, order='F')
    if Q is None:
        Q = np.zeros((n, n), dtype=float, order='F')
    if L is None:
        L = np.zeros((n, m), dtype=float, order='F')

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    Q_copy = np.asfortranarray(Q.copy())
    R_copy = np.asfortranarray(R.copy())
    L_copy = np.asfortranarray(L.copy())

    G_out = np.zeros((n, n), dtype=float, order='F')

    if jobl == 'Z':
        G, oufact, info = _sb02mt(
            'G', jobl, fact, uplo, n, m, A_copy, B_copy, Q_copy, R_copy, L_copy, G_out
        )
        ipiv = np.array([], dtype=np.int32)
    else:
        A_out, B_out, Q_out, L_out, G, oufact, info = _sb02mt(
            'G', jobl, fact, uplo, n, m, A_copy, B_copy, Q_copy, R_copy, L_copy, G_out
        )
        A_copy, B_copy, Q_copy, L_copy = A_out, B_out, Q_out, L_out
        ipiv = np.array([], dtype=np.int32)

    _check_info(info, 'sb02mt')

    return A_copy, B_copy, Q_copy, R_copy, L_copy, ipiv, oufact, G


def sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
           A, E, B, Q, R, L, ldwork=None, tol=0.0):
    """Solve generalized Riccati equation (slycot-compatible wrapper).

    slycot API: rcondu, X, alfar, alfai, beta, S, T, U, iwarn =
                sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p, A, E, B, Q, R, L)
    slicot API: X, rcondu, alfar, alfai, beta, S, T, U, iwarn, info =
                sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p, A, E, B, Q, R, L, tol)

    Returns
    -------
    rcondu : float
        Reciprocal condition number.
    X : ndarray
        Solution matrix.
    alfar, alfai, beta : ndarray
        Generalized eigenvalues.
    S : ndarray
        Schur form.
    T : ndarray
        Triangular factor.
    U : ndarray
        Orthogonal transformation.
    iwarn : int
        Warning indicator.
    """
    from slicot import sg02ad as _sg02ad

    A_copy = np.asfortranarray(A.copy())
    E_copy = np.asfortranarray(E.copy())
    B_copy = np.asfortranarray(B.copy())
    Q_copy = np.asfortranarray(Q.copy())
    R_copy = np.asfortranarray(R.copy())
    L_copy = np.asfortranarray(L.copy())

    X, rcondu, alfar, alfai, beta, S, T, U, iwarn, info = _sg02ad(
        dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
        A_copy, E_copy, B_copy, Q_copy, R_copy, L_copy, tol
    )

    _check_info(info, 'sg02ad', warn_codes=[1, 2])

    return rcondu, X, alfar, alfai, beta, S, T, U, iwarn


def sb01bd(n, m, np_, alpha, A, B, w, dico, tol=0.0, ldwork=None):
    """Pole placement via Varga method (slycot-compatible wrapper).

    slycot API: A_z, w_out, nfp, nap, nup, F, Z = sb01bd(n, m, np, alpha, A, B, w, dico)
    slicot API: a, wr, wi, nfp, nap, nup, F, Z, iwarn, info =
                sb01bd(dico, n, m, np, alpha, a, b, wr, wi, tol)

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Input dimension.
    np_ : int
        Number of eigenvalues to assign.
    alpha : float
        Threshold for fixed eigenvalues.
    A : ndarray
        State matrix.
    B : ndarray
        Input matrix.
    w : ndarray
        Desired eigenvalues (complex array).
    dico : str
        'C' for continuous, 'D' for discrete.

    Returns
    -------
    A_z : ndarray
        Modified A matrix.
    w_out : ndarray
        Assigned eigenvalues (complex).
    nfp : int
        Number of fixed poles.
    nap : int
        Number of assigned poles.
    nup : int
        Number of uncontrollable poles.
    F : ndarray
        Feedback gain matrix.
    Z : ndarray
        Orthogonal transformation.
    """
    from slicot import sb01bd as _sb01bd

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    wr = np.asfortranarray(np.real(w).copy())
    wi = np.asfortranarray(np.imag(w).copy())

    a_out, wr_out, wi_out, nfp, nap, nup, F, Z, iwarn, info = _sb01bd(
        dico, n, m, np_, alpha, A_copy, B_copy, wr, wi, tol
    )

    _check_info(info, 'sb01bd', warn_codes=[1, 2, 3])

    w_out = wr_out + 1j * wi_out

    return a_out, w_out, nfp, nap, nup, F, Z


def sb10ad(n, m, np_, ncon, nmeas, gamma, A, B, C, D, ldwork=None,
           job=1, gtol=0.0, actol=0.0):
    """H-infinity controller synthesis (slycot-compatible wrapper).

    slycot API: Ak, Bk, Ck, Dk, Ac, Bc, Cc, Dc, rcond =
                sb10ad(n, m, np, ncon, nmeas, gamma, A, B, C, D)
    slicot API: Ak, Bk, Ck, Dk, Ac, Bc, Cc, Dc, gamma_out, rcond, info =
                sb10ad(job, n, m, np, ncon, nmeas, A, B, C, D, gamma, gtol, actol)

    Returns
    -------
    Ak, Bk, Ck, Dk : ndarray
        Controller state-space matrices.
    Ac, Bc, Cc, Dc : ndarray
        Closed-loop system matrices.
    rcond : ndarray
        Reciprocal condition numbers.
    """
    from slicot import sb10ad as _sb10ad

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())
    D_copy = np.asfortranarray(D.copy())

    Ak, Bk, Ck, Dk, Ac, Bc, Cc, Dc, gamma_out, rcond, info = _sb10ad(
        job, n, m, np_, ncon, nmeas, A_copy, B_copy, C_copy, D_copy, gamma, gtol, actol
    )

    _check_info(info, 'sb10ad')

    return gamma_out, Ak, Bk, Ck, Dk, Ac, Bc, Cc, Dc, rcond


def sb10hd(n, m, np_, ncon, nmeas, A, B, C, D, ldwork=None, tol=0.0):
    """H2 controller synthesis (slycot-compatible wrapper).

    slycot API: Ak, Bk, Ck, Dk, rcond = sb10hd(n, m, np, ncon, nmeas, A, B, C, D)
    slicot API: Ak, Bk, Ck, Dk, rcond, info = sb10hd(n, m, np, ncon, nmeas, A, B, C, D, tol)

    Returns
    -------
    Ak, Bk, Ck, Dk : ndarray
        Controller state-space matrices.
    rcond : ndarray
        Reciprocal condition numbers.
    """
    from slicot import sb10hd as _sb10hd

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())
    D_copy = np.asfortranarray(D.copy())

    Ak, Bk, Ck, Dk, rcond, info = _sb10hd(
        n, m, np_, ncon, nmeas, A_copy, B_copy, C_copy, D_copy, tol
    )

    _check_info(info, 'sb10hd')

    return Ak, Bk, Ck, Dk, rcond


def ab08nd(n, m, p, A, B, C, D, equil='N', tol=0.0, ldwork=None):
    """Compute system zeros (slycot-compatible wrapper).

    slycot API: nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, Af, Bf =
                ab08nd(n, m, p, A, B, C, D, equil, tol)
    slicot API: nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, Af, Bf, info =
                ab08nd(equil, n, m, p, A, B, C, D, tol)

    Returns
    -------
    nu : int
        Number of finite zeros.
    rank : int
        Rank of system.
    dinfz : int
        Number of infinite zeros.
    nkror : int
        Number of right Kronecker indices.
    nkrol : int
        Number of left Kronecker indices.
    infz : ndarray
        Infinite zero structure.
    kronr : ndarray
        Right Kronecker indices.
    kronl : ndarray
        Left Kronecker indices.
    Af : ndarray
        Reduced A matrix.
    Bf : ndarray
        Reduced E matrix.
    """
    from slicot import ab08nd as _ab08nd

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())
    D_copy = np.asfortranarray(D.copy())

    nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, Af, Bf, info = _ab08nd(
        equil, n, m, p, A_copy, B_copy, C_copy, D_copy, tol
    )

    _check_info(info, 'ab08nd')

    return nu, rank, dinfz, nkror, nkrol, infz, kronr, kronl, Af, Bf


def ab09ad(dico, job, equil, n, m, p, A, B, C, nr=0, tol=0.0, ldwork=None):
    """Model reduction via balanced truncation (slycot-compatible wrapper).

    slycot API: Nr, Ar, Br, Cr, hsv = ab09ad(dico, job, equil, n, m, p, A, B, C, nr, tol)
    slicot API: ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(dico, job, equil, ordsel, n, m, p, nr, a, b, c, tol)

    Returns
    -------
    Nr : int
        Order of reduced system.
    Ar, Br, Cr : ndarray
        Reduced system matrices.
    hsv : ndarray
        Hankel singular values.
    """
    from slicot import ab09ad as _ab09ad

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())

    ordsel = 'A' if nr == 0 else 'F'

    Ar_full, Br_full, Cr_full, hsv, Nr_out, iwarn, info = _ab09ad(
        dico, job, equil, ordsel, n, m, p, nr, A_copy, B_copy, C_copy, tol
    )

    _check_info(info, 'ab09ad', warn_codes=[1])

    Ar = Ar_full[:Nr_out, :Nr_out].copy()
    Br = Br_full[:Nr_out, :].copy()
    Cr = Cr_full[:, :Nr_out].copy()

    return Nr_out, Ar, Br, Cr, hsv


def ab09md(dico, job, equil, n, m, p, A, B, C, alpha=0.0, nr=0, tol=0.0, ldwork=None):
    """Model reduction for unstable systems (slycot-compatible wrapper).

    slycot API: Nr, Ar, Br, Cr, Ns, hsv = ab09md(dico, job, equil, n, m, p, A, B, C, alpha, nr, tol)
    slicot API: ar, br, cr, ns, hsv, nr_out, iwarn, info = ab09md(dico, job, equil, ordsel, n, m, p, nr, alpha, a, b, c, tol)

    Returns
    -------
    Nr : int
        Order of reduced system.
    Ar, Br, Cr : ndarray
        Reduced system matrices.
    Ns : int
        Number of stable eigenvalues.
    hsv : ndarray
        Hankel singular values.
    """
    from slicot import ab09md as _ab09md

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())

    ordsel = 'A' if nr == 0 else 'F'

    Ar_full, Br_full, Cr_full, Ns, hsv, Nr_out, iwarn, info = _ab09md(
        dico, job, equil, ordsel, n, m, p, nr, alpha, A_copy, B_copy, C_copy, tol
    )

    _check_info(info, 'ab09md', warn_codes=[1])

    Ar = Ar_full[:Nr_out, :Nr_out].copy()
    Br = Br_full[:Nr_out, :].copy()
    Cr = Cr_full[:, :Nr_out].copy()

    return Nr_out, Ar, Br, Cr, Ns, hsv


def ab09nd(dico, job, equil, n, m, p, A, B, C, D, alpha=0.0, nr=0,
           tol1=0.0, tol2=0.0, ldwork=None):
    """Model reduction with DC matching (slycot-compatible wrapper).

    slycot API: Nr, Ar, Br, Cr, Dr, Ns, hsv =
                ab09nd(dico, job, equil, n, m, p, A, B, C, D, alpha, nr, tol1, tol2)
    slicot API: ar, br, cr, dr, nr_out, ns, hsv, iwarn, info =
                ab09nd(dico, job, equil, ordsel, n, m, p, nr, alpha, a, b, c, d, tol1, tol2)

    Returns
    -------
    Nr : int
        Order of reduced system.
    Ar, Br, Cr, Dr : ndarray
        Reduced system matrices.
    Ns : int
        Number of stable eigenvalues.
    hsv : ndarray
        Hankel singular values.
    """
    from slicot import ab09nd as _ab09nd

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())
    D_copy = np.asfortranarray(D.copy())

    ordsel = 'A' if nr == 0 else 'F'

    Ar_full, Br_full, Cr_full, Dr_full, Nr_out, Ns, hsv, iwarn, info = _ab09nd(
        dico, job, equil, ordsel, n, m, p, nr, alpha, A_copy, B_copy, C_copy, D_copy, tol1, tol2
    )

    _check_info(info, 'ab09nd', warn_codes=[1, 2])

    Ar = Ar_full[:Nr_out, :Nr_out].copy()
    Br = Br_full[:Nr_out, :].copy()
    Cr = Cr_full[:, :Nr_out].copy()
    Dr = Dr_full.copy()

    return Nr_out, Ar, Br, Cr, Dr, Ns, hsv


def ab13bd(dico, jobn, n, m, p, A, B, C, D, tol=0.0, ldwork=None):
    """Compute H2 or L2 norm (slycot-compatible wrapper).

    slycot API: norm = ab13bd(dico, jobn, n, m, p, A, B, C, D, tol)
    slicot API: norm, nq, iwarn, info = ab13bd(dico, jobn, A, B, C, D, tol)

    Returns
    -------
    norm : float
        The H2 or L2 norm.
    """
    from slicot import ab13bd as _ab13bd

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())
    D_copy = np.asfortranarray(D.copy())

    norm, nq, iwarn, info = _ab13bd(
        dico, jobn, A_copy, B_copy, C_copy, D_copy, tol
    )

    _check_info(info, 'ab13bd')

    return norm


def ab13dd(dico, jobe, equil, jobd, n, m, p, A, E, B, C, D, tol=0.0, ldwork=None):
    """Compute L-infinity norm (slycot-compatible wrapper).

    slycot API: gpeak, fpeak = ab13dd(dico, jobe, equil, jobd, n, m, p, A, E, B, C, D, tol)
    slicot API: gpeak, fpeak, info = ab13dd(dico, jobe, equil, jobd, n, m, p, fpeak_in, A, E, B, C, D, tol)

    Returns
    -------
    gpeak : float
        The L-infinity norm.
    fpeak : float
        Frequency at which peak occurs.
    """
    from slicot import ab13dd as _ab13dd

    A_copy = np.asfortranarray(A.copy())
    E_copy = np.asfortranarray(E.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())
    D_copy = np.asfortranarray(D.copy())
    fpeak_in = np.array([0.0, 1.0], order='F', dtype=float)

    gpeak, fpeak_out, info = _ab13dd(
        dico, jobe, equil, jobd, n, m, p, fpeak_in, A_copy, E_copy, B_copy, C_copy, D_copy, tol
    )

    _check_info(info, 'ab13dd')

    return gpeak[0], fpeak_out[0]


def ab13md(A, ITYPE, NBLOCK, ldwork=None):
    """Compute structured singular value (mu) (slycot-compatible wrapper).

    This is used for disk margin computations.

    slycot API: mu, D, G = ab13md(A, ITYPE, NBLOCK)
    slicot API: mu, D, G, info = ab13md(A, ITYPE, NBLOCK)

    Parameters
    ----------
    A : ndarray
        Complex matrix (frequency response at a single frequency).
    ITYPE : ndarray
        Integer array specifying block types.
    NBLOCK : ndarray
        Integer array specifying block sizes.

    Returns
    -------
    mu : float
        Upper bound on structured singular value.
    D : ndarray
        D scaling matrix.
    G : ndarray
        G scaling matrix.
    """
    from slicot import ab13md as _ab13md

    A_copy = np.asfortranarray(A.copy())
    ITYPE_copy = np.asfortranarray(ITYPE.astype(np.int32))
    NBLOCK_copy = np.asfortranarray(NBLOCK.astype(np.int32))

    bound, D, G, x, info = _ab13md(A_copy, ITYPE_copy, NBLOCK_copy)

    _check_info(info, 'ab13md')

    return bound, D, G


def tb01pd(n, m, p, A, B, C, job='M', equil='S', tol=0.0, ldwork=None):
    """Minimal realization (slycot-compatible wrapper).

    slycot API: Ar, Br, Cr, nr = tb01pd(n, m, p, A, B, C, job, equil, tol)
    slicot API: a, b, c, nr, nblk, info = tb01pd(job, equil, a, b, c, tol)

    Note: slicot tb01pd infers dimensions from array shapes.

    Returns
    -------
    Ar : ndarray
        Reduced A matrix.
    Br : ndarray
        Reduced B matrix.
    Cr : ndarray
        Reduced C matrix.
    nr : int
        Order of minimal realization.
    """
    from slicot import tb01pd as _tb01pd

    # Extract actual-sized arrays (caller may pass pre-padded arrays)
    A_copy = np.asfortranarray(A[:n, :n].copy())
    B_copy = np.asfortranarray(B[:n, :m].copy())
    C_copy = np.asfortranarray(C[:p, :n].copy())

    if tol is None:
        tol = 0.0

    Ar_full, Br_full, Cr_full, nr, nblk, info = _tb01pd(
        job, equil, A_copy, B_copy, C_copy, tol
    )

    _check_info(info, 'tb01pd')

    Ar = Ar_full[:nr, :nr].copy()
    Br = Br_full[:nr, :].copy()
    Cr = Cr_full[:, :nr].copy()

    return Ar, Br, Cr, nr


def _tb04ad_n1_fallback(m, p, A, B, C, D):
    """Fallback for tb04ad when n=1 (scalar state).

    For n=1: T(s) = C * B / (s - a) + D where a = A[0,0]
    Each output has the same denominator (s - a).
    """
    a = A[0, 0]

    # Denominator: s - a = [1, -a] (high to low coefficients)
    # All outputs share this denominator
    index = np.ones(p, dtype=np.int32)  # degree 1 for each output
    dcoeff = np.zeros((p, 2), dtype=float, order='F')
    for i in range(p):
        dcoeff[i, :] = [1.0, -a]

    # Numerator for T[i,j]: C[i,:] @ B[:,j] + D[i,j] * (s - a)
    # = D[i,j]*s + (C[i,:] @ B[:,j] - D[i,j]*a)
    # = [D[i,j], C[i,:] @ B[:,j] - D[i,j]*a]
    CB = C @ B
    ucoeff = np.zeros((p, m, 2), dtype=float, order='F')
    for i in range(p):
        for j in range(m):
            ucoeff[i, j, 0] = D[i, j]
            ucoeff[i, j, 1] = CB[i, j] - D[i, j] * a

    # Return controllable realization (same as input for n=1)
    return A.copy(), B.copy(), C.copy(), 1, index, dcoeff, ucoeff


def tb04ad(n, m, p, A, B, C, D, tol1=0.0, tol2=0.0, ldwork=None):
    """State-space to transfer function (slycot-compatible wrapper).

    slycot API: A_ctrb, B_ctrb, C_ctrb, nctrb, index, dcoeff, ucoeff =
                tb04ad(n, m, p, A, B, C, D, tol1)
    slicot API: a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info =
                tb04ad(rowcol, a, b, c, d, tol1, tol2)

    Returns
    -------
    A_ctrb : ndarray
        Transformed A matrix (controllable realization).
    B_ctrb : ndarray
        Transformed B matrix.
    C_ctrb : ndarray
        Transformed C matrix.
    nctrb : int
        Order of controllable part.
    index : ndarray
        Degrees of the denominator polynomials per output.
    dcoeff : ndarray
        Denominator polynomial coefficients.
    ucoeff : ndarray
        Numerator polynomial coefficients.
    """
    from slicot import tb04ad as _tb04ad

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())
    D_copy = np.asfortranarray(D.copy())

    # slicot's tb04ad has a bug when n=1 and m >= 3 (returns info=-24)
    # Use fallback for this case
    if n == 1 and m >= 3:
        return _tb04ad_n1_fallback(m, p, A_copy, B_copy, C_copy, D_copy)

    a_out, b_out, c_out, d_out, nr, index, dcoeff, ucoeff, info = _tb04ad(
        'R', A_copy, B_copy, C_copy, D_copy, tol1, tol2
    )

    _check_info(info, 'tb04ad')

    return a_out, b_out, c_out, nr, index, dcoeff, ucoeff


def tb05ad(n, m, p, jomega, A, B, C, job='NG', ldwork=None):
    """Frequency response evaluation (slycot-compatible wrapper).

    slycot API: (depends on job)
                job='NG': at, bt, ct, g, hinvb
                job='NH': g, hinvb
    slicot API: g, rcond, a_hess, b_trans, c_trans, info =
                tb05ad(baleig, inita, A, B, C, freq)

    Returns
    -------
    Depends on job parameter.
    """
    from slicot import tb05ad as _tb05ad

    A_copy = np.asfortranarray(A.copy())
    B_copy = np.asfortranarray(B.copy())
    C_copy = np.asfortranarray(C.copy())

    # Map slycot job parameter to slicot parameters:
    # job='NG' -> inita='G' (general A, compute Hessenberg)
    # job='NH' -> inita='H' (A already in Hessenberg form)
    baleig = 'N'
    inita = 'G' if job == 'NG' else 'H'

    g, rcond, a_hess, b_trans, c_trans, info = _tb05ad(
        baleig, inita, A_copy, B_copy, C_copy, jomega
    )

    _check_info(info, 'tb05ad')

    # hinvb = inv(jomega*I - A) * B is not returned by slicot
    # but it's not actually used by callers, so return None
    hinvb = None

    if job == 'NG':
        # Return input arrays as "transformed" matrices for compatibility
        # slicot's tb05ad doesn't return properly shaped transformed matrices
        return A_copy, B_copy, C_copy, g, hinvb, info
    else:
        return g, hinvb, info


def _convert_col_to_row_common_den(m, p, index_c, dcoeff_c, ucoeff_c):
    """Convert column-based common denominators to row-based.

    For 'C' mode: each column (input) has its own common denominator.
    For 'R' mode: each row (output) has its own common denominator.

    This function computes row-based data from column-based data by finding
    the product of all column denominators for each row (conservative LCM).
    """
    from numpy.polynomial import polynomial as P

    # For each output row, compute product of all input column denominators
    # (This is a conservative approximation of LCM that always works)
    # Coefficients in numpy.polynomial format: [const, x, x^2, ...]
    row_dens = []

    for i in range(p):
        lcm_poly = np.array([1.0])
        for j in range(m):
            deg_j = index_c[j]
            den_j = dcoeff_c[j, :deg_j + 1]
            den_j_rev = den_j[::-1]
            lcm_poly = P.polymul(lcm_poly, den_j_rev)
        row_dens.append(lcm_poly)

    # First pass: compute all adjusted numerators and find max degree
    adjusted_nums = []
    max_num_deg = 0

    for i in range(p):
        row_adjusted = []
        row_den = row_dens[i]
        for j in range(m):
            deg_j = index_c[j]
            col_den_j = dcoeff_c[j, :deg_j + 1]
            col_den_j_rev = col_den_j[::-1]
            quot, _ = P.polydiv(row_den, col_den_j_rev)

            # Numerator has same degree as column denominator (right-padded format)
            num_ij = ucoeff_c[i, j, :deg_j + 1]
            if np.abs(num_ij).max() < 1e-15:
                row_adjusted.append(None)
                continue
            num_ij_rev = num_ij[::-1]  # low to high for numpy.polynomial
            new_num = P.polymul(num_ij_rev, quot)
            row_adjusted.append(new_num)
            max_num_deg = max(max_num_deg, len(new_num) - 1)
        adjusted_nums.append(row_adjusted)

    # Compute kdcoef_r as max of denominator and numerator degrees
    max_den_deg = max(len(rd) - 1 for rd in row_dens)
    kdcoef_r = max(max_den_deg, max_num_deg) + 1

    # Build R-mode arrays
    index_r = np.zeros(p, dtype=np.int32)
    dcoeff_r = np.zeros((p, kdcoef_r), dtype=float, order='F')
    ucoeff_r = np.zeros((p, m, kdcoef_r), dtype=float, order='F')

    for i in range(p):
        row_den = row_dens[i]
        deg_i = len(row_den) - 1
        index_r[i] = deg_i
        dcoeff_r[i, :deg_i + 1] = row_den[::-1]

        for j in range(m):
            new_num = adjusted_nums[i][j]
            if new_num is None:
                continue
            new_num_rev = new_num[::-1]  # high to low
            deg_new = len(new_num_rev) - 1  # actual polynomial degree
            # Right-align: place coefficients to match denominator indexing
            # For proper TF, deg_new <= deg_i, so pad with leading zeros
            start_idx = deg_i - deg_new
            ucoeff_r[i, j, start_idx:deg_i + 1] = new_num_rev

    return index_r, dcoeff_r, ucoeff_r


def td04ad(rowcol, m, p, index, dcoeff, ucoeff, tol=0.0, ldwork=None):
    """Transfer function to state-space (slycot-compatible wrapper).

    slycot API: nr, A, B, C, D = td04ad(rowcol, m, p, index, dcoeff, ucoeff, tol)
    slicot API: nr, A, B, C, D, info = td04ad(rowcol, m, p, index, dcoeff, ucoeff, tol)

    Parameters
    ----------
    rowcol : str
        'R' for rows over common denominators, 'C' for columns.
    m : int
        Number of system inputs.
    p : int
        Number of system outputs.
    index : ndarray
        Degrees of denominators (length m for 'C', length p for 'R').
    dcoeff : ndarray
        Denominator coefficients (m x kdcoef for 'C', p x kdcoef for 'R').
    ucoeff : ndarray
        Numerator coefficients (p x m x kdcoef).

    Returns
    -------
    nr : int
        Order of the resulting state-space system.
    A, B, C, D : ndarray
        State-space matrices.
    """
    from slicot import td04ad as _td04ad

    # ucoeff may be padded to square; trim to (p, m, kdcoef)
    ucoeff_trimmed = ucoeff[:p, :m, :]

    # slicot's td04ad has issues with rowcol='C' when p != m (non-square)
    # Work around by converting to 'R' mode
    if rowcol == 'C' and p != m:
        index_r, dcoeff_r, ucoeff_r = _convert_col_to_row_common_den(
            m, p, index, dcoeff, ucoeff_trimmed
        )
        index_copy = np.asfortranarray(index_r, dtype=np.int32)
        dcoeff_copy = np.asfortranarray(dcoeff_r)
        ucoeff_copy = np.asfortranarray(ucoeff_r)
        rowcol_actual = 'R'
    else:
        index_copy = np.asfortranarray(index.copy(), dtype=np.int32)
        dcoeff_copy = np.asfortranarray(dcoeff.copy())
        ucoeff_copy = np.asfortranarray(ucoeff_trimmed.copy())
        rowcol_actual = rowcol

    nr, A, B, C, D, info = _td04ad(
        rowcol_actual, m, p, index_copy, dcoeff_copy, ucoeff_copy, tol
    )

    _check_info(info, 'td04ad')

    return nr, A, B, C, D


def mb03rd(n, A, X, pmax=1.0, tol=0.0, ldwork=None):
    """Block diagonal Schur form (slycot-compatible wrapper).

    slycot API: Aout, Xout, blsize, w = mb03rd(n, A, X, pmax)
    slicot API: a, x, nblcks, blsize, wr, wi, info =
                mb03rd(jobx, sort, a, pmax, x, tol)

    Returns
    -------
    Aout : ndarray
        Block diagonal Schur form.
    Xout : ndarray
        Transformation matrix.
    blsize : ndarray
        Block sizes.
    w : ndarray
        Eigenvalues (complex).
    """
    from slicot import mb03rd as _mb03rd

    A_copy = np.asfortranarray(A.copy())
    X_copy = np.asfortranarray(X.copy())

    Aout, Xout, nblcks, blsize, wr, wi, info = _mb03rd(
        'U', 'N', A_copy, pmax, X_copy, tol
    )

    _check_info(info, 'mb03rd')

    w = wr + 1j * wi

    return Aout, Xout, blsize[:nblcks], w


# If using slycot (not slicot), overwrite with direct imports from slycot
if _use_slycot and not _use_slicot:
    from slycot import (  # noqa: F811
        sb03od, sb04md, sb04qd, sg03ad,
        sb02md, sb02mt, sg02ad, sb01bd,
        sb10ad, sb10hd, ab08nd,
        ab09ad, ab09md, ab09nd,
        ab13bd, ab13dd, ab13md,
        tb01pd, tb04ad, tb05ad, td04ad, mb03rd,
    )
    from slycot.exceptions import (  # noqa: F811
        SlycotResultWarning as SlicotResultWarning,
        SlycotArithmeticError as SlicotArithmeticError,
    )

    from slycot import sb03md57

    def sb03md(n, C, A, U, dico, job='X', fact='N', trana='N', ldwork=None):  # noqa: F811
        """Wrapper for slycot's sb03md57."""
        ret = sb03md57(A, U, C, dico, job, fact, trana, ldwork)
        return ret[2:]
