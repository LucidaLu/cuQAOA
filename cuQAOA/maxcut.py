from scipy.optimize import fmin_bfgs as BFGS
from tqdm import tqdm
from .cu_qaoa_sim import cuQAOASim
from numba import cuda
import numpy as np
from .cu_kernels import *
from loguru import logger

from enum import Enum


class GraphType(Enum):
    ERDOS_RENYI = 0,
    THREE_REGULAR = 1,
    UNSPECIFIED = 2


def linear_args(p, T):
    samples = (np.arange(1, p + 1) - 0.5) / p
    dt = T / p
    gamma = samples * dt
    beta = (1 - samples) * dt
    return np.concatenate((beta, gamma))


def maxcut_H_C(G):
    n = G.number_of_nodes()
    H_C = cuda.device_array(1 << n, dtype=np.int16)
    edges = np.array(G.edges, dtype=np.int32)

    @ cuda.jit
    def compute_cut_kernel(a):
        s = cuda.grid(1)
        a[s] = 0
        for x, y in edges:
            a[s] += 1 if s >> x & 1 != s >> y & 1 else -1
            # critical! not ZZ, it's I-ZZ
    compute_cut_kernel[get_grid(1 << n)](H_C)
    return H_C


def solve_maxcut(G, p, gtype=GraphType.UNSPECIFIED, state=False):
    '''
        return maxcut expectation achieved by QAOA
        if state = True, (expectation, the statevector) 
    '''
    qaoa = cuQAOASim(maxcut_H_C(G))

    def optimization(x):
        res = BFGS(lambda x: qaoa(*np.split(x, 2)), x, full_output=1, disp=0)
        return (res[0], (len(G.edges) - res[1]) / 2) if state else (len(G.edges) - res[1]) / 2

    if gtype == GraphType.ERDOS_RENYI:
        return optimization(linear_args(p, p * 0.55))
    elif gtype == GraphType.THREE_REGULAR:
        return optimization(linear_args(p, p * 0.75))
    else:
        return max(optimization(linear_args(p, p * dt)) for dt in tqdm(np.arange(0.1, 1, 0.1)))


if __name__ == '__main__':
    import random as rnd
    import networkx as nx

    cuda.select_device(1)
    SEED = 1
    n = 12
    rnd.seed(SEED)
    np.random.seed(SEED)
    logger.info(SEED)
    G = nx.random_regular_graph(n=n, d=3)
    # qw = cuQAOASim(maxcut_H_C(G))
    # print(qw(np.linspace(1, 2, 5), np.linspace(2, 3, 5)))
    logger.info(len(G.edges))
    logger.info(solve_maxcut(G, 8, gtype=GraphType.THREE_REGULAR))
    logger.info(solve_maxcut(G, 8, gtype=GraphType.UNSPECIFIED))
