import numpy as np


def uniform_quantizer(in_val: np.ndarray, n_bits: int, x_max: float, m: int) -> np.ndarray:
    """
    Quantizes an input signal using a uniform quantizer.

    Parameters:
        in_val (ndarray): Input sample(s) to be quantized.
        n_bits (int): Number of bits for quantization (defines the number of levels, L=2**n_bits).
        x_max (float): Maximum absolute value that can be represented.
        m (int): Offset parameter; m = 0 for midrise and m = 1 for midtread quantizer.

    Returns:
        ndarray: Quantization indices corresponding to the input samples.
    """
    L = 2**n_bits
    delta = 2 * x_max / L
    offset = m * delta / 2
    

    in_val = np.clip(in_val, -x_max, x_max)
    
  
    q_ind = np.floor((in_val + x_max - offset) / delta).astype(int)
    
 
    q_ind = np.clip(q_ind, 0, L - 1)
    
    return q_ind

def uniform_dequantizer(q_ind: np.ndarray, n_bits: int, x_max: float, m: int) -> np.ndarray:
    """
    Dequantizes an input signal using a uniform dequantizer.

    Parameters:
        q_ind (ndarray): Quantization indices to be dequantized.
        n_bits (int): Number of bits for quantization (defines the number of levels, L=2**n_bits).
        x_max (float): Maximum absolute value that can be represented.
        m (int): Offset parameter; m = 0 for midrise and m = 1 for midtread quantizer.

    Returns:
        ndarray: Dequantized values corresponding to the quantization indices.
    """
    L = 2**n_bits
    delta = 2 * x_max / L
    offset = m * delta / 2
    

    deq_val =  q_ind * delta - x_max + offset + delta / 2
    

    deq_val = np.clip(deq_val, -x_max, x_max)
    
    return deq_val

def expand(signal: np.ndarray, m: float):
    """
    Expand signal to make it uniform
    Parameters: 
        signal (np.ndarray): signal to be compressed
        m (float): (miew) the compression factor

    Returns:
        np.ndarray: compressed signal
    """
    return (((1 + m) ** np.abs(signal)) - 1) / m

# (((1 + m) ** np.abs(signal)) - 1) / m
# (np.log(1 + m * signal) / np.log(1 + m))
def compress(signal: np.ndarray, m: float):
    """
    Compress signal back
    Parameters: 
        signal (np.ndarray): signal to be compressed
        m (float): (miew) the compression factor

    Returns:
        np.ndarray: compressed signal
    """
    return  (np.log(1 + m * np.abs(signal)) / np.log(1 + m))
