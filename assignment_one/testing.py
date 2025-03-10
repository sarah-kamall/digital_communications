import numpy as np
import matplotlib.pyplot as plt
from processing import *


def deterministic_test():

    x = np.arange(-6, 6.01, 0.01)
    n_bits = 3
    xmax = 6

    m = 0
    q_ind_m0 = uniform_quantizer(x, n_bits, xmax, m)
    deq_val_m0 = uniform_dequantizer(q_ind_m0, n_bits, xmax, m)

    plt.figure(figsize=(10, 6))
    # Plot the input ramp and dequantized signal on the primary y-axis
    plt.plot(x, x, label='Input Signal (Ramp)', color='blue')
    plt.step(x, deq_val_m0, label='Dequantized Signal (m=0)',
             color='red', where='mid')
    plt.xlabel('Input Value')
    plt.ylabel('Signal Value')
    plt.title('Quantizer/Dequantizer Output (m=0, Midrise)')
    plt.grid(True)

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(x, q_ind_m0, label='Quantization Indices (m=0)',
             color='green', alpha=0.6)
    ax2.set_ylabel('Quantization Index')
    ax2.set_yticks(np.arange(min(q_ind_m0), max(q_ind_m0) + 1, 1))  # Integer quantization indices

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.show()

    # === For m = 1 (midtread quantizer) ===
    m = 1
    q_ind_m1 = uniform_quantizer(x, n_bits, xmax, m)
    deq_val_m1 = uniform_dequantizer(q_ind_m1, n_bits, xmax, m)

    plt.figure(figsize=(10, 6))

    plt.plot(x, x, label='Input Signal (Ramp)', color='blue')
    plt.step(x, deq_val_m1, label='Dequantized Signal (m=1)',
             color='red', where='mid')
    plt.xlabel('Input Value')
    plt.ylabel('Signal Value')
    plt.title('Quantizer/Dequantizer Output (m=1, Midtread)')
    plt.grid(True)

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(x, q_ind_m1, label='Quantization Indices (m=1)',
             color='green', alpha=0.6)
    ax2.set_ylabel('Quantization Index')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.show()


def random_test():
    uniform_random_variables = np.random.uniform(-5, 5, 10000)
    x_max = 5
    snr_sim_db = []
    snr_theory_db = []
    n_bits_values = np.arange(2, 9)
    input_power = x_max**2 / 3

    for n in n_bits_values:
        q_sample = uniform_quantizer(uniform_random_variables, n, x_max, 0)
        deq_sample = uniform_dequantizer(q_sample, n, 5, 0)
        error_power = np.mean((uniform_random_variables - deq_sample) ** 2)
        snr_simulation_db = 10 * np.log10(input_power / error_power)
        snr_sim_db.append(snr_simulation_db)
        delta = (2 * x_max) / pow(2, n)
        error_power = (delta ** 2) / 12
        snr_theoritical_db = 10 * np.log10(input_power / error_power)
        snr_theory_db.append(snr_theoritical_db)
    plt.figure(figsize=(10, 6))
    plt.plot(n_bits_values, snr_sim_db, 'bo-', label='Simulated SNR (dB)')
    plt.plot(n_bits_values, snr_theory_db,
             'rs--', label='Theoretical SNR (dB)')
    plt.xlabel('Number of bits (n_bits)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs. Number of Bits for Uniform Quantization (m=0, xmax=5)')
    plt.grid(True)
    plt.legend()
    plt.show()


def random_nonuniform_test():
    signs = np.random.choice([-1, 1], size=10000)
    magnitudes = np.random.exponential(scale=1, size=10000)
    non_uniform_random_variables = signs * magnitudes
    x_max = 5
    snr_sim_db = []
    n_bits_values = np.arange(2, 9)

    for n in n_bits_values:
        q_sample = uniform_quantizer(non_uniform_random_variables, n, x_max, 0)
        deq_sample = uniform_dequantizer(q_sample, n, 5, 0)
        error_power = np.mean((non_uniform_random_variables - deq_sample) ** 2)
        input_power = np.mean(non_uniform_random_variables ** 2)
        snr_simulation_db = 10 * np.log10(input_power / error_power)
        snr_sim_db.append(snr_simulation_db)

    plt.figure(figsize=(10, 6))
    plt.plot(n_bits_values, snr_sim_db, 'bo-', label='Simulated SNR (dB)')
    plt.xlabel('Number of bits (n_bits)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs. Number of Bits for Uniform Quantization (m=0, xmax=5)')
    plt.grid(True)
    plt.legend()
    plt.show()


def random_nonuniform_with_compression():
    # Generate non-uniform random variables
    signs = np.random.choice([-1, 1], size=10000)
    magnitudes = np.random.exponential(scale=1, size=10000)
    non_uniform_random_variables = signs * magnitudes

    # Normalization step (before compression)
    x_max = np.max(np.abs(non_uniform_random_variables))  
    normalized_signal = non_uniform_random_variables / x_max  # Now in range [-1,1]

    n_bits_values = np.arange(2, 9)
    mu_values = [0, 5, 100, 200]
    SNR_mu = np.zeros((len(n_bits_values), len(mu_values)))

    for j, m in enumerate(mu_values):
        for i, n in enumerate(n_bits_values):
            # Expand (if mu > 0)
            expanded_signal = signs * expand(np.abs(normalized_signal), m) if m != 0 else normalized_signal  
            
            # Quantization
            q_sample = uniform_quantizer(expanded_signal, n, 1, 0)  # Adjusted x_max to 1
            deq_sample = uniform_dequantizer(q_sample, n, 1, 0)  # Adjusted x_max to 1

            # Compress (if mu > 0)
            compressed_signal = signs * compress(np.abs(deq_sample), m) if m != 0 else deq_sample

            # Denormalization step (after expansion)
            final_signal = compressed_signal * x_max  # Scale back to original range

            # Compute error and SNR
            error_power = np.mean((non_uniform_random_variables - final_signal) ** 2)
            input_power = np.mean(non_uniform_random_variables ** 2)
            snr_simulation_db = 10 * np.log10(input_power / error_power)
            SNR_mu[i][j] = snr_simulation_db

    # Plot SNR vs n_bits
    plt.figure()
    markers = ['bo-', 'r*-', 'gs-', 'kd-']
    for j, mu in enumerate(mu_values):
        plt.plot(n_bits_values, SNR_mu[:, j], markers[j], label=f'μ={mu}')
    plt.xlabel('n_bits')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs n_bits for μ-law Quantization (Non-uniform Input)')
    plt.legend()
    plt.show()


deterministic_test()
random_test()
random_nonuniform_test()
random_nonuniform_with_compression()
