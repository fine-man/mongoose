import torch
import sys

sys.path.append('/home/vanshg/play/iiith/research-cvit/mongoose/mongoose_slide')
from mongoose_slide.slide_lib.cupy_kernel import cupyKernel

import numpy as np
import cupy as cp

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

kernel_code = '''
extern "C" __global__
void binaryToDecimal(const int* input, int N, int L, int K, int* output) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (tid_x < L && tid_y < N) {
        int decimal_value = 0;
        for (int i = 0; i < K; ++i) {
            int bit = input[tid_y * L * K + tid_x * K + i];
            decimal_value = (decimal_value << 1) | bit;
        }
        output[tid_y * L + tid_x] = decimal_value;
    }
}
'''

# Compile the kernel
module = cp.RawModule(code=kernel_code)

# Create a function wrapper for the CUDA kernel
binary_to_decimal_kernel = module.get_function('binaryToDecimal')

class RandomProjection:
    def __init__(self, 
                 D, # feature dimension of data
                 K, # Number of bits per data-point
                 L # Number of hash tables
    ):
        """
        Initialize the RandomProjection class.

        Args:
            D (int): Feature dimension of data.
            K (int): Number of bits per data-point.
            L (int): Number of hash tables.
        """
        self.D = D
        self.K = K
        self.L = L
        
        self.plane_norms = torch.randn((D, L * K)).to(device)
        print(self.plane_norms.device)
        self.fp = cupyKernel(kernel_code, "binaryToDecimal")

    def hash(self, data, transpose=False):
        """
        Hash the input data.

        Args:
            data (torch.Tensor): Torch tensor of size (N, D).
            transpose (bool, optional): Whether to transpose the data. Defaults to False.

        Returns:
            torch.Tensor: Torch tensor of size (N, L) containing the L hash tables for the N data-points.
        """
        N, D = data.shape[:2]
        # (N, LK) = (N, D) @ (D, LK)
        srp = torch.matmul(data.to(device), self.plane_norms) # (N, LK)
        srp = (srp > 0).to(torch.int32)
        srp = srp.reshape(N, self.L, self.K)

        result = self.binary_to_decimal(srp)
        return result

    def binary_to_decimal(self, srp):
        """
        Convert binary values to decimal values.

        Args:
            srp (torch.Tensor): Input tensor of size (N, L, K).

        Returns:
            torch.Tensor: Output tensor of size (N, L) containing the decimal values.
        """
        # Input tensor should be a PyTorch tensor
        assert isinstance(srp, torch.Tensor)

        # Transfer PyTorch tensor to GPU
        input_array_gpu = cp.asarray(srp.cpu().numpy())

        # Get dimensions of the input tensor
        N, L, K = srp.size()

        # Create an output array to store decimal values
        output_array = cp.zeros((N, L), dtype=cp.int32)

        # Set block and grid dimensions
        block_dim = (32, 32)
        grid_dim = ((L + block_dim[0] - 1) // block_dim[0], (N + block_dim[1] - 1) // block_dim[1])

        # Launch the CUDA kernel
        binary_to_decimal_kernel(grid_dim, block_dim, args=(input_array_gpu, N, L, K, output_array))

        output_np = cp.asnumpy(output_array)
        output_tensor = torch.from_numpy(output_np).to(device)

        return output_tensor
    
    def fingerprint(self, srp):
        """
        Generate fingerprints for the input data.

        Args:
            srp (torch.Tensor): Torch tensor of size (N, L, K).

        Returns:
            torch.Tensor: Torch tensor of size (N, L) containing the L hash tables for the N data-points.
        """
        N = srp.shape[0]
        result = torch.zeros(N, self.L, dtype=torch.int32)
        # Set block and grid dimensions
        block_dim = (32, 32)
        grid_dim = ((self.L + block_dim[0] - 1) // block_dim[0], (N + block_dim[1] - 1) // block_dim[1])

        self.fp(grid=grid_dim,
                block=block_dim,
                args=[srp.data_ptr(), self.K, self.L, result.data_ptr()],
                strm=torch.cuda.current_stream().cuda_stream)

        return result.int()