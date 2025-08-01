import numpy as np
from lascar import TraceBatchContainer, Session, CpaEngine
from lascar.tools.aes import sbox
from numba import njit
import scipy.stats

@njit
def hamming(x, y):
    xor = x ^ y
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance

def moving_average_filter(data, window_size=3):
    # Create a moving average (boxcar) window
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def standardize_traces(traces):
    mean = np.mean(traces, axis=0)
    std_dev = np.std(traces, axis=0)
    standardized_traces = (traces - mean) / std_dev
    return standardized_traces

def preprocess_traces(traces):
    # Apply moving average filter
    filtered_traces = np.array([moving_average_filter(trace) for trace in traces])
    # Standardize traces
    standardized_traces = standardize_traces(filtered_traces)
    return standardized_traces

def print_statistics(traces, label):
    print(f"Statistics for {label}:")
    print(f"  Mean: {np.mean(traces):.5f}")
    print(f"  Std Deviation: {np.std(traces):.5f}")
    print(f"  Min value: {np.min(traces):.5f}")
    print(f"  Max value: {np.max(traces):.5f}")
    print(f"  Skewness: {scipy.stats.skew(traces.flatten()):.5f}")
    print(f"  Kurtosis: {scipy.stats.kurtosis(traces.flatten()):.5f}")

def generate_guess_function(byte, reference_state=0x00):
    def guess_function(values, guess):
        return hamming(sbox[values[byte] ^ guess], reference_state)
    return guess_function

if __name__ == '__main__':

    plain_text = np.load('../traces/plaintext_AES_SW_1000.npy')
    traces = np.load('../traces/traces_AES_SW_1000.npy')

    print_statistics(traces, "Raw Traces")

    # Preprocess the traces
    traces = preprocess_traces(traces)

    print_statistics(traces, "Preprocess Traces")

    # Setup for Correlation Power Analysis
    batch = TraceBatchContainer(traces, plain_text)
    guess_range = range(0, 256)
    engines = [CpaEngine(f"cpa_{byte}", generate_guess_function(byte), guess_range) for byte in range(16)]

    # Run the CPA session
    session = Session(batch, engines=engines)
    session.run(batch_size=100)

    # Retrieve and print the guessed keys
    keys = np.zeros(16)
    for i, engine in enumerate(engines):
        keys[i] = engine.finalize().max(1).argmax()
    print("Keys: " + ','.join([hex(int(k)) for k in keys]))
