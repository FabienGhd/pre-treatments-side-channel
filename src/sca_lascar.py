from lascar import *
from lascar.tools.aes import sbox

import matplotlib.pyplot as plt


def generate_guess_function(byte):
    def guess_function(values, guess):
        return hamming(sbox[values[byte] ^ guess])
    return guess_function

if __name__=='__main__':
    # retrieve data
    plain_text = np.load('../traces/plaintext_AES_SW_1000.npy')
    traces = np.load('../traces/traces_AES_SW_1000.npy')

    # Initialize the container
    batch = TraceBatchContainer(traces, plain_text)
    batch.leakage_processing = None

    guess_range = range(0,256)

    # use hamming weight on the byte i as a selection function
    engines = [
            CpaEngine(
                f"cpa_{byte}", 
                generate_guess_function(byte), 
                guess_range
                ) 
            for byte in range(0, 16)
            ]


    session = Session(
            batch, 
            engines = engines,
            )
    session.run(batch_size = 1000)

    keys = np.zeros(16);
    for i in range(0, len(engines)):
        keys[i] = np.abs(engines[i].finalize()).max(1).argmax()
    
    print("Keys: " + ','.join([hex(int(k)) for k in keys]))
