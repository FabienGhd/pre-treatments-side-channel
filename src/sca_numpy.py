import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

NB_CANDIDATES = 256
NB_TRACES = 4000

Sbox = np.array([
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

############################################################################
def hamming_weight(data, byte_id, k):
    """
    hamming_weight
    Compute the hamming_weight
    Args:
        - data: a N x 16 np.uint8 values
        - byte_id: a np.uint8 value between 0 to 15. It's the byte to consider
        - k: the key candidate
    Returns:
        - a N np.uint8 array : the hamming weight of data[:,byte_id]
    """
    data = data[:,byte_id]
    x = Sbox[k ^ data]
    res = np.zeros(x.shape[0])
    # Hamming weight
    for i, d in enumerate(x):
        res[i] = d.bit_count()
    return res

############################################################################
def hamming_distance(data, byte_id, k):
    """
    hamming_distance(data, byte_id, k)
    Compute the hamming distance
    Args:
        - data: a N x 16 np.uint8 values
        - byte_id: a np.uint8 value between 0 to 15. It's the byte to consider
        - k: the key candidate
    Returns:
        - a N np.uint8 array : the hamming distance between data[:,byte_id] and the old value
          of data[:,byte]
    """
    rev_shift_row = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
    
    res = np.zeros(data.shape[0], dtype=np.uint8)
    old_data = np.zeros(data.shape[0], dtype=np.uint8)

    for i, arr in enumerate(data):
        old_data[i] = arr[rev_shift_row[byte_id]]

    data_byte = np.uint8(data[:,byte_id])
    data_byte = ReverseSBox[k ^ data_byte]
    data_byte = data_byte ^ old_data
    
    for i, b in enumerate(data_byte):
        res[i] = b.bit_count()

    return res

############################################################################
def basis_function_w(p, values, byte_id, k):
    """
    basis_function
    Compute the p-th basis function on the value: value
    Args:
        - p: a np.uint8 value corresponding to the number of the basis function
        - value: a N x 16 np.uint8 value to compute
    Returns:
        - a N x 16 np.uint8 : the result of pth function on data values
    """
    if (p <= 0):
        return 1 # constant function
    values = np.uint8(values[:, byte_id])
    values = Sbox[values ^ k]
    return (values >> (p - 1)) & 1

############################################################################
def basis_function_d(p, values, byte_id, k):
    """
    basis_function
    Compute the p-th basis function on the value: value
    Args:
        - p: a np.uint8 value corresponding to the number of the basis function
        - values: a N x 16 np.uint8 values to compute
    Returns:
        - a N x 16 uint8: results of pth func on data values
    """
    if (p <= 0):
        return 1 # constant function
    rev_shift_row = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
    
    old_values = np.zeros(values.shape[0], dtype=np.uint8)

    for i, arr in enumerate(values):
        old_values[i] = arr[rev_shift_row[byte_id]]

    values = np.uint8(values[:, byte_id])
    values = ReverseSBox[k ^ values]
    values = values ^ old_values
    return (values >> (p - 1)) & 1

############################################################################
def compute_power_model(plain_texts, func):
    """
    compute_power_model
    Compute the power model of plain_texts, using func as modele
    Args:
        - plain_texts: a matrices of N x b np.uint8
        - func: a function that take a np.uint8 in entry and return a np.uint8
    Return:
        - a array of 256 matrice of size N x b corresponding to the power model
          of each plaintext value for each key candidate
    """
    res = np.zeros((NB_CANDIDATES, NB_TRACES, len(plain_texts[0])))
    for k in range(0, NB_CANDIDATES):
        res[k] = np.stack(np.vectorize(func)(Sbox[plain_texts ^ k]))
    return res


############################################################################
def lra_find_key(data, leakage, byte_id, base_func=basis_function_w):
    """
    lra_find_key
    Compute a LRA attack on data and leakage to find a subkey

    Args:
        - data: an array of N x 16 np.uint8 values corresponding to the plaintext
        - leakage: a matrice of N x d np.uint8 values corresponding to the leakage
    Return:
        - an array of 256 floats corresponding to the least error for each key
    """
    # prepocessing leakage
    #L = np.zeros((np.unique(data).shape[0], leakage.shape[1]));
    L = leakage;
    #for i,x in enumerate(np.unique(data)):
    #    L[i] = np.mean(leakage[data == x], axis=0)
    # preprocessing data
    #D = np.unique(data)
    D = data[:,byte_id]

    # precompute SST
    SST = np.sum(L ** 2, axis = 0) - ((np.sum(L, axis = 0) ** 2) / L.shape[0])

    bests_candidate = np.zeros((NB_CANDIDATES))
    for k in range(NB_CANDIDATES):
        # compute M_k 
        M = np.zeros((D.shape[0], 9))
        for h in range(0, 9):
            M[:,h] = base_func(h, data, byte_id, k)

        P = np.linalg.inv((M.T).dot(M)).dot(M.T)

        # compute B
        B = P.dot(L)

        # compute estimator E
        E = M.dot(B)
        
        # compute R2 for B
        SSR = np.sum((E - L) ** 2)
        R2 = 1 - (SSR / SST)

        # retrieve the best candidate
        bests_candidate[k] = np.max(R2);

    return bests_candidate

############################################################################
def cpa_acc(data, leakage, byte_id, step=1, pw_mod=hamming_weight):
    """
    cpa_acc
    Compute a CPA attack on data and leakage by cumulation of N / step traces

    Args:
        - data: an array of N x 16 np.uint8 values. Each value is the x_i plaintext byte of n_i 
          traces
        - leakage: a matrice of N x d np.uint8 values. Each row is a leakage traces of d points
        - byte_id: the index of the plaintext to used
        - step: the size of the data use by each cumulation step
        - pw_mod: the function used to compute the power model
    Returns:
        - A matrice of (N / step) x 256 np.float values. Each row is an array of candidates with
          theirs correlation coefficient
    """
    # Initialisation of variables
    num_iteration = data.shape[0] // step
    res = np.zeros((num_iteration, 256))

    leak_acc = np.zeros(leakage.shape[1])
    leak_square_acc = np.zeros(leakage.shape[1])
    pow_mod_acc = np.zeros(256)
    pow_mod_square_acc = np.zeros(256)
    leak_pow_acc = np.zeros((256, leakage.shape[1]))

    # Compute num_iteration CPA
    for i in range(step, data.shape[0] + step, step):
        # Get working data
        data_cur = data[(i - step):i,byte_id]
        leak_cur = leakage[(i - step):i]
        
        # Leakage mean/var processing
        leak_acc += np.sum(leak_cur, axis = 0);
        leak_square_acc += np.sum(leak_cur ** 2, axis = 0)
        leak_mean = leak_acc / i
        leak_var = (leak_square_acc / i) - (leak_mean ** 2)

        # Hypotheses mean and variance processing
        pow_mod = np.zeros((256, step))
        for k in range(0, 256):
            pow_mod[k] = pw_mod(data[(i - step):i], byte_id, k)
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

def lra_step(data, traces, byte_id, step=1, pw_mod=basis_function_w):
    """
    lra_step
    Compute lra from step traces to len(traces) traces.
    Args:
        - data   : np.darray(N, dtype=uint8) plaintext
        - traces : np.darray((N, D), dtype=float) leakage traces
    Returns:
        - np.darray((N // step, 256), dtype=uint8) ranks of each key for each step 
    """
    num_iter = data.shape[0] // step
    res = np.zeros((num_iter, 256))
    for i in range(step, data.shape[0] + step, step):
        traces_cur = traces[:i]
        data_cur = data[:i]
        res[((i - step) // step)] = lra_find_key(data_cur, traces_cur, byte_id, base_func=pw_mod)
    return res

def ranks_evaluation(data, traces, keys, attack ,step=1, pw_mod=hamming_distance):
    """
    ranks_evaluation
    Compute the attack for each key and return the mean rank evolution
    Args:
        - data: np.darray((N, 16), dtype = uint8) the N plaintexts  
        - traces: np.darrat((N, D), dtype = double) the N leakages traces
        - keys: np.darray(16, dtype = uint8) the correct keys
        - attack: a function taking :
            data: np.darray(N)
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
    for i in range(4):
        key_rank = attack(data, traces, i, step=step, pw_mod=pw_mod)
        for j in range(num_step):
            res[j] += np.where(np.argsort(key_rank[j])[::-1] == keys[i])[0][0]
    res = res / 4;
    return res



if __name__ == '__main__':

    # get the traces and plain text
    file = h5py.File("../traces/aes_hd_ext.h5", 'r')

    cipher_text = file['Attack_traces']['metadata']['ciphertext'][:NB_TRACES]
    traces      = file['Attack_traces']['traces'][:NB_TRACES,900:]
    """
    plain_text = np.load('../traces/plaintext_AES_SW_1000.npy')
    plain_text = plain_text[:NB_TRACES]
    traces = np.load('../traces/traces_AES_SW_1000.npy')
    traces = traces[:NB_TRACES];
    """
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

    """
    correct_key = [
            0x2b, 0x7e, 0x15, 0x16, 
            0x28, 0xae, 0xd2, 0xa6, 
            0xab, 0xf7, 0x15, 0x88, 
            0x9 , 0xcf, 0x4f, 0x3c
            ]
    """
    correct_key = [
            208, 20, 249, 168, 
            201,238,37,137,
            225, 63, 12, 200, 
            182, 99, 12, 166
            ]

    step = 100
    # Test cpa attack
    file = open("hd_result.npy", "rb")

    plt.close("all")
    fig, ax = plt.subplots()
    for label in leakage.keys():
        tmp = np.load(file)
        ax.plot(range(step, NB_TRACES + step, step), tmp, label=label)
    ax.legend(loc="upper right")
    plt.show()

    plt.close("all")
    fig, ax = plt.subplots()
    for label in leakage.keys():
        tmp = np.load(file)
        ax.plot(range(step, NB_TRACES + step, step), tmp, label=label)
    ax.legend(loc="upper right")
    plt.show()


    #for label, leak in tqdm (leakage.items(), desc="Progress..."):
    #    rank = ranks_evaluation(
    #            cipher_text, leak,  correct_key, 
    #            cpa_acc, step=step, pw_mod=hamming_distance)
    #    np.save(file, rank);

        #ax.plot(range(step, NB_TRACES + step, step), rank, label=label)
    #ax.legend(loc="upper right")
    #plt.show()
    
    # Test LRA attack
    #plt.close("all")
    #fig, ax = plt.subplots()
    #for label, leak in tqdm (leakage.items(), desc="Progress..."):
    #    rank = ranks_evaluation(
    #            cipher_text, leak, correct_key, 
    #            lra_step, step=step, pw_mod=basis_function_d)
    #    np.save(file, rank);
    #    ax.plot(range(step, NB_TRACES + step, step), rank, label=label)
    #ax.legend(loc="upper right")
    #plt.show()
