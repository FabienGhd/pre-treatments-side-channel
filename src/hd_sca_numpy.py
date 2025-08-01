import numpy as np
import h5py

import matplotlib.pyplot as plt
from tqdm import tqdm 

NB_CANDIDATES = 256
NB_TRACES = 5000

SBox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
    , dtype = np.uint8) 




Rcon = np.array ([0x01, 0x2, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36], dtype = np.uint8)


ReverseSBox = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d], dtype=np.uint8)



"""
K0  key0,0 key0,1 key0,2 key0,3
    key1,0 key1,1 key1,2 key1,3
    key2,0 key2,1 key2,2 key2,3
    key3,0 key3,1 key3,2 key3,3


K1  SB-K0_1,3 + K0_0,0 + Rcon0  |  K1_0,0 + K0_0,1  |  K1_0,1 + K0_0,2  |  K1_0,2 + K0_0,3
    SB-K0_2,3 + K0_1,0          |  K1_1,0 + K0_1,1  |  K1_1,1 + K0_1,2  |  K1_1,2 + K0_1,3
    SB-K0_3,3 + K0_2,0          |  K1_2,0 + K0_2,1  |  K1_2,1 + K0_2,2  |  K1_2,2 + K0_2,3
    SB-K0_0,3 + K0_3,0          |  K1_3,0 + K0_3,1  |  K1_3,1 + K0_3,2  |  K1_3,2 + K0_3,3


Ki  SB-K(i-1)_1,3 + K(i-1)_0,0 + Rcon(i-1)  |  Ki_0,0 + K(i-1)_0,1  |  Ki_0,1 + K(i-1)_0,2  |  Ki_0,2 + K(i-1)_0,3
    SB-K(i-1)_2,3 + K(i-1)_1,0              |  Ki_1,0 + K(i-1)_1,1  |  Ki_1,1 + K(i-1)_1,2  |  Ki_1,2 + K(i-1)_1,3
    SB-K(i-1)_3,3 + K(i-1)_2,0              |  Ki_2,0 + K(i-1)_2,1  |  Ki_2,1 + K(i-1)_2,2  |  Ki_2,2 + K(i-1)_2,3
    SB-K(i-1)_0,3 + K(i-1)_3,0              |  Ki_3,0 + K(i-1)_3,1  |  Ki_3,1 + K(i-1)_3,2  |  Ki_3,2 + K(i-1)_3,3

================================================================

K(i-1)  Ki_0,0 + SB-K(i-1)_1,3 + Rcon(i-1)  |  Ki_0,1 + Ki_0,0  |  Ki_0,2 + Ki_0,1  |  Ki_0,3 + Ki_0,2
        Ki_1,0 + SB-K(i-1)_2,3              |  Ki_1,1 + Ki_1,0  |  Ki_1,2 + Ki_1,1  |  Ki_1,3 + Ki_1,2
        Ki_2,0 + SB-K(i-1)_3,3              |  Ki_2,1 + Ki_2,0  |  Ki_2,2 + Ki_2,1  |  Ki_2,3 + Ki_2,2
        Ki_3,0 + SB-K(i-1)_0,3              |  Ki_3,1 + Ki_3,0  |  Ki_3,2 + Ki_3,1  |  Ki_3,3 + Ki_3,2
"""
############################################################################
def reverse_key_expansion(k10):
    """
    reverse_key_expansion
    Computes K0 from K10

    forall k0, k0 = reverse_key_expansion( key_expansion(k0).last )

    Args:
      - k10: last of the 11 keys of AES-128

    Returns:
      - k0
    """
    state = to_square(k10)
    for i in range(10, 0, -1):
        tmp = np.zeros((4, 4), dtype=np.uint8)

        for c in range(3, 0, -1):
            tmp[:, c] = state[:, c] ^ state[:, c - 1]
        for r in range(4):
            tmp[r, 0] = state[r, 0] ^ SBox[tmp[(r + 1) & 3, 3]]

        tmp[0, 0] ^= Rcon[i - 1]
        state = tmp

    return from_square(state)



##############################################################################
def key_expansion (k0):
    """
    key_expansion
    Compute the 11 subkeys use in the AES-128.
    
    Args:
      - k0: main key (also the first key round), k0 have to be provide has an array 
        of 16 bytes (np.uint8)

    Returns:
      - K: list of the 11 subkeys in a square reprensentation. 
               K = [K [i]_{j, k}]_{j < 4, k < 4, i < 11}.
        K is a list. K [i] is the i-th round key in square representation
        a (4x4)-bytes array of np.uint8.
      
    """
    K = [to_square(k0)]
    for i in range(10):
        round_key = np.zeros((4, 4), dtype=np.uint8)
        round_key[0, 0] = SBox[K[i][1, 3]] ^ K[i][0, 0] ^ Rcon[i]
        round_key[1, 0] = SBox[K[i][2, 3]] ^ K[i][1, 0]
        round_key[2, 0] = SBox[K[i][3, 3]] ^ K[i][2, 0]
        round_key[3, 0] = SBox[K[i][0, 3]] ^ K[i][3, 0]
        for j in range(1, 4):
            round_key[:, j] = round_key[:, j - 1] ^ K[i][:, j]
        K.append(round_key)
    return K


############################################################################
def to_square  (inpt):
    """
    to_square
    Transform a 16 bytes array (np.uint8) to a 2-dimensions array of 
    4x4 bytes (np.uint8) (representation use for the internal state inside the AES)

    Args:
      - inpt: {inpt_i}_{i < 16} = [inpt_0, inpt_1, ..., inpt_15], with inpt_i a byte (np.uint8)
 
    Returns:
      - output: square represenation of inpt, {output_{j, k}}_{j < 4, k < 4}.
      
      inpt = [inpt_{0}, inpt_{1}, inpt_{2}, inpt_{3}, inpt_{4}, inpt_{5}, inpt_{6}, inpt_{7},
    inpt_{8}, inpt_{9}, inpt_{10}, inpt_{11}, inpt_{12}, inpt_{13}, inpt_{14}, inpt_{15},]

                            [[inpt_{0}, inpt_{4}, inpt_{8} , inpt_{12}]       
     to_square (inpt) =      [inpt_{1}, inpt_{5}, inpt_{9} , inpt_{13}]     
                             [inpt_{2}, inpt_{6}, inpt_{10}, inpt_{14}]      
                             [inpt_{3}, inpt_{7}, inpt_{11}, inpt_{15}]]      
     
                            [[output_{0, 0}, output_{0, 1}, output_{0, 2}, output_{0, 3}]
                      =      [output_{1, 0}, output_{1, 1}, output_{1, 2}, output_{1, 3}]
                             [output_{2, 0}, output_{2, 1}, output_{2, 2}, output_{2, 3}]
                             [output_{3, 0}, output_{3, 1}, output_{3, 2}, output_{3, 3}]]

    """
    return np.array(inpt, dtype=np.uint8).reshape((4, 4), order='F')


############################################################################
def from_square  (inpt):
    """
    from_square
    transform a 2-dimensions array of 4x4 bytes (np.uint8)
    (representation use for the internal state inside the AES) to an array 
    16 bytes (np.int8).

    Args:
      - inpt: square represenation use in the AES {inpt_{j, k}}_{j < 4, k < 4}. 
      with inpt_{i, j} a byte (np.uint8).

      [[inpt_{0, 0}, inpt_{0, 1}, inpt_{0, 2}, inpt_{0, 3}]
      [inpt_{1, 0}, inpt_{1, 1}, inpt_{1, 2}, inpt_{1, 3}]
      [inpt_{2, 0}, inpt_{2, 1}, inpt_{2, 2}, inpt_{2, 3}]
      [inpt_{3, 0}, inpt_{3, 1}, inpt_{3, 2}, inpt_{3, 3}]]
 
    Returns:
      - output: 
      
              [[inpt_{0, 0}, inpt_{0, 1}, inpt_{0, 2}, inpt_{0, 3}]
     inpt =    [inpt_{1, 0}, inpt_{1, 1}, inpt_{1, 2}, inpt_{1, 3}]
               [inpt_{2, 0}, inpt_{2, 1}, inpt_{2, 2}, inpt_{2, 3}]
               [inpt_{3, 0}, inpt_{3, 1}, inpt_{3, 2}, inpt_{3, 3}]]

                         [inpt_{0, 0}, inpt_{1, 0}, inpt_{2, 0}, inpt_{3, 0}, inpt_{0, 1}, 
    to_squqre (inpt) =    inpt_{1, 1}, inpt_{2, 1}, inpt_{3, 1}, inpt_{0, 2}, inpt_{1, 2}, 
                          inpt_{2, 2}, inpt_{3, 2}, inpt_{3, 0}, inpt_{3, 1}, inpt_{3, 2},
                          inpt_{3, 3}]

                     = [output [0], output [1], ... , output [15]]

    """
    return inpt.flatten(order='F')

############################################################################
def pearson_coeff (traces:np.array, Y:np.array) -> np.array:
############################################################################
    """pearson_coeff

    compute the pearson coefficient between the traces and all guessed
    distributions (output of time_leakage_model)

    Args:   
      - traces: D x Q-array of float-64bits (np.float64).

      - Y: (S x Q)-array of int-8bits (np.uint8).
      """

    D, _ = traces.shape
    S, _= Y.shape

    X = np.float64 (traces)
    Y = np.float64 (Y)

    nom_1 = X - X.mean (1) [:, None]
    nom_2 = Y - Y.mean (1) [:, None]

    nom   = np.dot (nom_1, nom_2.T)
    denom = np.sqrt (np.dot ((nom_1**2).sum (1).reshape (D, 1), (nom_2**2).sum (1).reshape (S, 1).T))

    return nom/denom


############################################################################
def hamming_weight(x):
    """
    hamming_weight
    Compute the hamming_weight of x
    Args:
        - x: a np.uint8 value
    Returns:
        - a np.uint8 : the number of One bits in x
    """
    return x.bit_count()


############################################################################
def correlation(cov, mean_a, mean_b, var_a, var_b):
    """
    correlation
    Computes the correlation between 2 arrays given their covariance, mean and variance. Used for CPA.

    Args:
      - cov: the covariance between the 2 arrays
      - mean_a: the mean of the first array
      - mean_b: the mean of the second array
      - var_a: the variance of the first array
      - var_b: the variance of the second array

    Returns:
      - the correlation between the 2 arrays
    """

    low_part = np.sqrt(var_a * var_b)
    high_part = cov - mean_a * mean_b
    return high_part / low_part


############################################################################
def compute_power_model(ciphertext):
    """
    compute_power_model
    Computes the power model of ciphertext, using hamming distance.

    Args:
      - ciphertext: a matrices of N x 16 np.uint8

    Return:
      - a array of 256 matrices of size N x 16 corresponding to the power model
        of each ciphertext byte for each key candidate
    """

    res = np.zeros((NB_CANDIDATES, NB_TRACES, len(ciphertext[0])))

    rev_shiftrow_idx = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]    # [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]

    length = len(ciphertext)

    for k in tqdm (range(0, NB_CANDIDATES)):
        # tmp = np.zeros((NB_TRACES, len(ciphertext[0])), dtype=np.uint8)
        for i in range(NB_TRACES):
            for j in range(16):
                # jth slot content before last turn SBox & shift row occured
                old_byte = np.uint8(ciphertext[i][j])
                # jth slot content after last turn SBox & shift row occured
                new_byte = np.uint8(ciphertext[i][rev_shiftrow_idx [j]])

                res[k, i, j] = hamming_weight (ReverseSBox[k ^ old_byte] ^ new_byte)
    
    return res


############################################################################
def find_key(i, powers_models, traces):
    """
    find_key
    Computes a CPA attack to find a byte of the key

    Args:
      - i: the index of the byte we are looking for. in [0, 15]
      - power_models: the power model computed with `compute_power_model`
      - traces: the leakage

    Returns:
      - a 256 x 16 array of the guess made by the CPA for the `i`th byte of the key.
        Should be filtered with argmax afterwards.
    """
    pw = np.zeros((NB_CANDIDATES, NB_TRACES))
    for k in range(0, NB_CANDIDATES):
        pw[k] = powers_models[k][:,i]

    mean_k = np.mean(pw, axis=1)
    var_k = np.var(pw, axis=1)
    mean_trace = np.mean(traces, axis=0)
    var_trace = np.var(traces, axis=0)

    cov = pw.dot(traces) / NB_TRACES
    res = np.zeros((NB_CANDIDATES, len(traces[0])))
    for k in range(0, NB_CANDIDATES):
        res[k] =  np.abs(correlation(cov[k], mean_trace, mean_k[k], var_trace, var_k[k]))
    return res


############################################################################
def basis_function(p, value):
    """
    basis_function
    Compute the p-th basis function on the value: value
    Args:
        - p: a np.uint8 value corresponding to the number of the basis function
        - value: a np.uint8 value to compute
    Returns:
        - a uint8
    """
    if (p <= 0):
        return 1 # constant function
    return (value >> (p - 1)) & 1


# byte_index: the current byte being determined out of 16
# data: the cipher text
# leakage: the traces
############################################################################
def lra_find_key(byte_index, data, leakage):
    """
    lra_find_key
    Compute a Hardware LRA attack on cipher text and leakage to find a byte of the key

    Args:
        - byte_index: the index of current subkey we are looking for
        - data: an array of N x 16 np.uint8 values corresponding to the ciphertext
        - leakage: a matrice of N x d np.uint8 values corresponding to the leakage
    Return:
        - the best guess for the `byte_index` th byte of the key
    """

    # bytes at byte_index in all 16 chunks in data
    data_byte_index = data[:, byte_index]

    # preprocessing data
    D = np.unique(data_byte_index)

    # prepocessing leakage
    L = np.zeros((D.shape[0], leakage.shape[1]));
    for i,x in enumerate(D):
        L[i] = np.mean(leakage[data_byte_index == x], axis=0)



    # precompute SST
    SST = np.sum(L ** 2, axis=0) - (np.sum(L, axis = 0) ** 2) / L.shape[1] 


    rev_shiftrow_idx =  [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]    # [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]

    # compute M_k for each candidates
    M = np.zeros((NB_CANDIDATES, D.shape[0], 9))
    for k in range(NB_CANDIDATES):
        for c in range(D.shape[0]):
            for h in range(0, 9):
                # byte_indexth slot content before last turn SBox & shift row occured
                old_byte = np.uint8(data[c][byte_index])
                # byte_indexth slot content after last turn SBox & shift row occured
                new_byte = np.uint8(data[c][rev_shiftrow_idx[byte_index]])

                M[k][c][h] = basis_function(h, ReverseSBox[k ^ old_byte] ^ new_byte)

    # compute B_k for each candidates
    B = np.zeros((NB_CANDIDATES, 9, L.shape[1]))
    for k in range(NB_CANDIDATES):
        B[k] = np.linalg.inv(M[k].T.dot(M[k])).dot(M[k].T).dot(L)

    # compute estimator E_k for each candidates
    E = np.zeros((NB_CANDIDATES, D.shape[0], L.shape[1]))
    for k in range(NB_CANDIDATES):
        E[k] = M[k].dot(B[k])
    
    # compute R2 for each B_k
    R2 = np.zeros((NB_CANDIDATES, L.shape[1]))
    for k in range(NB_CANDIDATES):
        SSR = np.sum((E[k] - L) ** 2)
        R2[k] = 1 - (SSR / SST)

    # retrieve the best candidate
    bests_candidate = np.zeros((NB_CANDIDATES))
    for k in range(NB_CANDIDATES):
        bests_candidate[k] = np.max(R2[k]);

    return np.argmax(bests_candidate)



############################################################################
def cpa_acc(all_data, index, leakage, step=1, pw_mod=hamming_weight):
    """
    cpa_acc
    Compute a CPA attack on data and leakage by cumulation of N / step traces

    Args:
        - all_data: an array of N x 16 np.uint8 values.
        - index: the current byte index being determined, in [0, 16[
        - leakage: a matrice of N x d np.uint8 values. Each row is a leakage traces of d points
        - step: the size of the data use by each cumulation step
        - pw_mod: the function used to compute the power model
    Returns:
        - A matrice of (N / step) x 256 np.float values. Each row is an array of candidates with
          theirs correlation coefficient
    """

    data = all_data[:, index]

    # Initialisation of variables
    num_iteration = data.shape[0] // step
    res = np.zeros((num_iteration, 256))

    leak_acc = np.zeros(leakage.shape[1])
    leak_square_acc = np.zeros(leakage.shape[1])
    pow_mod_acc = np.zeros(256)
    pow_mod_square_acc = np.zeros(256)
    leak_pow_acc = np.zeros((256, leakage.shape[1]))

    rev_shiftrow_idx = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]

    # Compute num_iteration CPA
    for i in range(step, data.shape[0] + step, step):
        # Get working data
        data_cur = data[(i - step):i]
        leak_cur = leakage[(i - step):i]
        
        # Leakage mean/var processing
        leak_acc += np.sum(leak_cur, axis = 0);
        leak_square_acc += np.sum(leak_cur ** 2, axis = 0)
        leak_mean = leak_acc / i
        leak_var = (leak_square_acc / i) - (leak_mean ** 2)


        # Hypotheses mean and variance processing
        pow_mod = np.zeros((256, step))
        for k in range(0, 256):
            for j in range(step):
                old_byte = np.uint8(data_cur[j])
                new_byte = np.uint8(all_data[i - step + j][rev_shiftrow_idx[index]])
                pow_mod[k][j] = hamming_weight( ReverseSBox[k ^ old_byte] ^ new_byte )

        pow_mod_acc += np.sum(pow_mod, axis = 1)
        pow_mod_square_acc += np.sum(pow_mod ** 2, axis = 1)
        pow_mod_mean = pow_mod_acc / i
        pow_mod_var = (pow_mod_square_acc / i) - (pow_mod_mean ** 2)

        # Correlations processing
        correlations = np.zeros((256, leak_cur.shape[1]))
        for k in range(0, 256):
            leak_pow_acc[k] += pow_mod[k].dot(leak_cur)
            correlations[k] = np.abs(
                    ((leak_pow_acc[k] / i) - (leak_mean * pow_mod_mean[k])) /
                    (np.sqrt(pow_mod_var[k] * leak_var))
                    )

        for k in range(0, 256):
            res[(i - step) // step][k] = np.max(correlations[k])
    return res


def ranks_evaluation(data, traces, keys, attack ,step=1):
    """
    ranks_evaluation
    Compute the attack for each key and return the mean rank evolution
    Args:
        - data: np.darray((N, 16), dtype = uint8) the N plaintexts  
        - traces: np.darrat((N, D), dtype = double) the N leakages traces
        - keys: np.darray(16, dtype = uint8) the correct keys
        - attack: a function taking :
            data: np.darray((N, 16))
            i: int
            traces: np.darray((N,D))
            step: uint
          and returning:
            np.darray((N // step, 256))
        - step: uint the number of traces added each step
    Returns:
        - np.darray(N / step, dtype = float) the mean evolution of the rank
    """
    num_step = data.shape[0] // step
    res = np.zeros(num_step, dtype = float)
    for i in range(16):
        key_rank = attack(data, i, traces, step=step)
        for j in range(num_step):
            res[j] += np.where(np.argsort(key_rank[j])[::-1] == keys[i])[0][0]
    res = res / 16;
    return res




if __name__ == '__main__':


    with h5py.File('../traces/aes_hd_ext.h5', 'r') as file:
        cipher_text = file['Attack_traces']['metadata']['ciphertext'][:NB_TRACES]
        plain_text  = file['Attack_traces']['metadata']['plaintext'][:NB_TRACES]
        traces      = file['Attack_traces']['traces'][:NB_TRACES]

        # Compute pre-process
        leakage = {
            "raw":          traces,
            "square":       traces ** 2,
            "abs":          np.abs(traces),
            "centered":     np.zeros(traces.shape),
            "standardized": np.zeros(traces.shape),
        }
        for (i, trace) in enumerate(traces):
            leakage["centered"][i] = trace - np.mean(trace)
            leakage["standardized"][i] = leakage["centered"][i] / np.var(trace)


        correct_key = [
            0x2b, 0x7e, 0x15, 0x16, 
            0x28, 0xae, 0xd2, 0xa6, 
            0xab, 0xf7, 0x15, 0x88, 
            0x9 , 0xcf, 0x4f, 0x3c
        ]

        k10 = from_square(key_expansion(correct_key)[10])
        print(f"k10 is {k10}")

        power_models = compute_power_model (cipher_text)

        """
        #PEARSON coeff
        for b in range (0, 16):
            c = pearson_coeff (traces [:, 900:].T, power_models [:, :, b])
            print (c.shape)
            plt.plot (np.abs (c), color = 'g', alpha = 0.4)
            plt.plot (np.abs (c [:, k10 [b]]), color = 'r', alpha = 1)
            
            plt.show ()
            
            max_correlations = np.zeros(256)
            for k in range(256):
                max_correlations[k] = np.max(c[:, k])
            resulting_byte = np.argmax(max_correlations)
            print(f"pearson coef k10[{b}] {resulting_byte}")
        """


        """
        #LRA
        result = np.zeros(16, dtype=np.uint8)

        for i in range(16):
            result[i] = lra_find_key(i, cipher_text, traces)

        print(f"lra k10={result}")
        """

        """
        # CPA
        result = np.zeros(16, dtype=np.uint8)

        for i in range(0, len(plain_text[0])):
            correlations = find_key(i, power_models, traces)

            max_correlations = np.zeros(NB_CANDIDATES)
            for k in range(NB_CANDIDATES):
                max_correlations[k] = np.max(correlations[k])
            result[i] = np.argmax(max_correlations)

        print(f"cpa k10 is {result}")
        """

        fig, ax = plt.subplots()
        for label, leak in tqdm (leakage.items(), desc="Progress..."):
            rank = ranks_evaluation(cipher_text, leak, correct_key, cpa_acc)
            ax.plot(range(1, NB_TRACES + 1, 1), rank, label=label)
        ax.legend(loc="upper right")
        plt.show()