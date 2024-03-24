import numpy as np
import scipy

def cmdct(x, odd=True):
    """ Calculate complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:True.

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(x) // 2
    n0 = (N + 1) / 2
    if odd:
        outlen = N
        pre_twiddle = np.exp(-1j * np.pi * np.arange(N * 2) / (N * 2))
        offset = 0.5
    else:
        outlen = N + 1
        pre_twiddle = 1.0
        offset = 0.0

    post_twiddle = np.exp(
        -1j * np.pi * n0 * (np.arange(outlen) + offset) / N
    )

    X = scipy.fftpack.fft(x * pre_twiddle)[:outlen]

    if not odd:
        X[0] *= np.sqrt(0.5)
        X[-1] *= np.sqrt(0.5)

    return X * post_twiddle * np.sqrt(1 / N)



def mdct(x, odd=True):
    """ Calculate modified discrete cosine transform of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:True.

    Returns
    -------
    out : array_like
        The output signal

    """
    return np.real(cmdct(x, odd=odd)) * np.sqrt(2)