import numpy as np


def uniform_quantizer(in_val: np.ndarray, n: int, x_max: float, m: int) -> np.ndarray:
    """
    Quantizes an input signal using a uniform quantizer.

    Parameters:
        in_val(ndarray): Input sample(s) to be quantized.
        n_bits(int): Number of bits for quantization(defines the number of levels, L=2**n_bits).
        xmax(float): Maximum absolute value that can be represented.
        m(int): Offset parameter; m = 0 for midrise and m = 1 for midtread quantizer.

    Returns:
        ndarray: Quantization indices corresponding to the input samples.
    """
    L = pow(2, n)
    delta = 2 * x_max / L
    offset = m * delta / 2
    quantized_values = np.round((in_val - offset) // delta)
    return quantized_values


def uniform_dequantizer(q_val: np.ndarray, n: int, x_max: float, m: int) -> np.ndarray:
    """
    Dequantizes an input signal using a uniform dequantizer.

    Parameters:
        q_val(ndarray): Input sample(s) to be dequantized.
        n_bits(int): Number of bits for quantization(defines the number of levels, L=2**n_bits).
        xmax(float): Maximum absolute value that can be represented.
        m(int): Offset parameter; m = 0 for midrise and m = 1 for midtread quantizer.

    Returns:
        ndarray: Dequantization indices corresponding to the quantized samples.
    """
    L = pow(2, n)
    delta = 2 * x_max / L
    offset = m * delta / 2
    org_values = np.round(q_val * delta + offset)
    org_values = np.clip(org_values, -x_max, x_max)
    return org_values


def expand(signal: np.ndarray, m: float):
    """
    Expand signal to make it uniform
    Parameters: 
        signal (np.ndarray): signal to be compressed
        m (float): (miew) the compression factor

    Returns:
        np.ndarray: compressed signal
    """
    return (np.log10(1 + m * signal) / np.log10(1 + m))


def compress(signal: np.ndarray, m: float):
    """
    Compress signal back
    Parameters: 
        signal (np.ndarray): signal to be compressed
        m (float): (miew) the compression factor

    Returns:
        np.ndarray: compressed signal
    """
    return (((1 + m) ** signal) - 1) / m
