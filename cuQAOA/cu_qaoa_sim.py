import numpy as np
from numba import cuda
from .cu_kernels import *
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def apply_dc_sum(s):
    '''
        Destructively sum the whole array.
        The result will be stored at s[0].
    '''
    n = len(s).bit_length() - 1
    for i in range(n):
        t = min(PTHREAD_MAX, 1 << (n - i - 1))
        b = (1 << (n - i - 1)) // t
        binary_add_kernel[b, t](s, i)


def fast_walsh_hadamard_transform(a):
    n = len(a).bit_length() - 1
    for h in range(n):
        fwht_step_kernel[get_grid(1 << (n - 1))](a, h)


class cuQAOASim:
    def __init__(self, H_C):
        '''
            Compute the highest energy of H_C.
            In QAOA this is done by computing the ground energy of -H_C.
        '''
        n = self.n = len(H_C).bit_length() - 1
        self.H_C = H_C if cuda.is_cuda_array(H_C) else cuda.to_device(H_C)
        self.grid = get_grid(1 << self.n)

        @ cuda.jit
        def compute_x_sum_kernel(s):
            x = cuda.grid(1)
            s[x] = 0
            for i in range(n):
                s[x] += -1 if x >> i & 1 else 1

        self.H_Bc = cuda.device_array(2**n, dtype=np.int8)
        compute_x_sum_kernel[self.grid](self.H_Bc)
        self.state = cuda.device_array(2**n, dtype=np.complex128)

    def compute_qaoa_state(self, x):
        # putting kernel function inside results in repeated compiling
        assign_constant_kernel[self.grid](self.state, 1 / 2**(self.n / 2))
        p = len(x) // 2
        for i in range(p):
            b, g = x[i], x[i + p]
            phase_shift_kernel[self.grid](self.state, -1j * g, self.H_C)
            fast_walsh_hadamard_transform(self.state)
            phase_shift_kernel[self.grid](self.state, -1j * b, self.H_Bc)
            fast_walsh_hadamard_transform(self.state)
            multiply_by_constant_kernel[self.grid](self.state, 1. / (1 << self.n))

    def __call__(self, beta, gamma):
        x = np.concatenate((beta, gamma))
        self.compute_qaoa_state(x)
        compute_expectation_kernel[get_grid(1 << self.n)](self.state, self.H_C)
        apply_dc_sum(self.state)
        return -np.real(self.state[0])
