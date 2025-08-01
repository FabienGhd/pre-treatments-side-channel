import numpy as np
from lascar import TraceBatchContainer, Session

# Loading the traces
traces_path = '../traces/traces_AES_SW_1000.npy'
traces = np.load(traces_path)

# Loading the plaintexts
plaintexts_path = '../traces/plaintext_AES_SW_1000.npy'
plaintexts = np.load(plaintexts_path)

# Printing the shapes to understand the dataset
print(f"Traces shape: {traces.shape}")
print(f"Plaintexts shape: {plaintexts.shape}")

"""
The dataset consists of 1000 traces, each with 5000 data points, and 1000 corresponding plaintexts, each 16 bytes long.
Basically, we have 1000 different traces for analysis, with each trace capturing 5000 measurements.
The plaintexts are what was encrypted, with each entry being 16 bytes, aligning with the AES block size.
"""

container = TraceBatchContainer(traces, plaintexts)

session = Session(container)

session.run()