from numba import cuda
import cmath


@cuda.jit
def fwht_step_kernel(a, h):
    i = cuda.grid(1)
    x, y = i >> h, i & ((1 << h) - 1)
    x = x << (h + 1) | y
    y = x + (1 << h)
    a[x], a[y] = a[x] + a[y], a[x] - a[y]


@cuda.jit
def multiply_by_constant_kernel(s, k):
    i = cuda.grid(1)
    s[i] *= k


@cuda.jit
def assign_constant_kernel(s, k):
    i = cuda.grid(1)
    s[i] = k


@cuda.jit
def phase_shift_kernel(s, k, p):
    i = cuda.grid(1)
    s[i] = cmath.exp(k * p[i]) * s[i]


@cuda.jit
def diff_phase_shift_kernel(s, k, p):
    i = cuda.grid(1)
    s[i] = -1j * p[i] * cmath.exp(k * p[i]) * s[i]


@cuda.jit
def compute_expectation_kernel(s, h):
    i = cuda.grid(1)
    s[i] = s[i].conjugate() * h[i] * s[i]


@cuda.jit
def compute_expectation_2_kernel(s0, h, s1):
    i = cuda.grid(1)
    s0[i] = s0[i].conjugate() * h[i] * s1[i]


@cuda.jit
def binary_add_kernel(s, w):
    i = cuda.grid(1)
    s[i << (w + 1)] += s[((i << 1) + 1) << w]


@cuda.jit
def element_access_kernel(s, v):
    i = cuda.grid(1)
    v[i] = s[i]


PTHREAD_MAX = 1024


def get_grid(dim):
    PTHREAD = min(PTHREAD_MAX, dim)
    NBLOCK = dim // PTHREAD
    return (NBLOCK, PTHREAD)
