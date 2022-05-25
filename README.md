# cuQAOA: A CUDA-based QAOA Simulator

**cuQAOA** is a CUDA-based QAOA Simulator, which implements state vector simulation of QAOA using Fast Walsh–Hadamard Transform.

In this project, we implement CUDA programs with the help of [Numba](https://numba.pydata.org/) library. So in order to use this project, you need to have numba installed following [Numba offical documentation](https://numba.readthedocs.io/en/stable/user/installing.html), and make sure CUDA GPU support for Numba is enabled.

## Requirement
```
numpy
scipy
numba
networkx
```

## Usage

### cuQAOASim
A class that is initialized with an array-like problem Hamiltonian $H_C$, and `cuQAOASim` will try to find the **highest energy** of $H_C$ (or equivalently, the ground energy of $-H_C$).

After initialized, you can invoke `__call__` method with two arrays $\beta,\gamma$ that computes the expectation of $-H_C$ when QAOA arguments are configured $\beta,\gamma$.

```py
from cuQAOA import cuQAOASim

from networkx import nx

N = 12
G = nx.erdos_renyi_graph(n=N, p=0.8)

cq = cuQAOASim([sum(1 if s >> x & 1 != s >> y & 1 else -1 for x, y in G.edges) for s in range(1 << N)])

print('random argument qaoa result =', (len(G.edges) - cq([1, 2, 3, 4], [5, 6, 7, 8])) / 2)
```

### solve_maxcut
A wrapper for solving maxcut with QAOA.
```py
from cuQAOA.maxcut import solve_maxcut, GraphType

from networkx import nx

N = 12
G = nx.erdos_renyi_graph(n=N, p=0.8)

print('optimized argument qaoa result =', solve_maxcut(G, 4, gtype=GraphType.ERDOS_RENYI))
```