import numpy as np
import scipy

def imdct(X, odd=True):
    """ Calculate inverse modified discrete cosine transform of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return icmdct(X, odd=odd) * np.sqrt(2)



def icmdct(X, odd=True):
    """ Calculate inverse complex MDCT/MCLT of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    if not odd and len(X) % 2 == 0:
        raise ValueError(
            "Even inverse CMDCT requires an odd number "
            "of coefficients"
        )

    X = X.copy()

    if odd:
        N = len(X)
        n0 = (N + 1) / 2

        post_twiddle = np.exp(
            1j * np.pi * (np.arange(N * 2) + n0) / (N * 2)
        )

        Y = np.zeros(N * 2, dtype=X.dtype)
        Y[:N] = X
        Y[N:] = -1 * np.conj(X[::-1])
    else:
        N = len(X) - 1
        n0 = (N + 1) / 2

        post_twiddle = 1.0

        X[0] *= np.sqrt(2)
        X[-1] *= np.sqrt(2)

        Y = np.zeros(N * 2, dtype=X.dtype)
        Y[:N+1] = X
        Y[N+1:] = -1 * np.conj(X[-2:0:-1])

    pre_twiddle = np.exp(1j * np.pi * n0 * np.arange(N * 2) / N)

    y = scipy.fftpack.ifft(Y * pre_twiddle)

    return np.real(y * post_twiddle) * np.sqrt(N)