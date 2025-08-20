import torch
import numpy as np
import sys
import os

try:
    import paddle
    has_paddle = True
except ImportError:
    has_paddle = False

sys.path.append("../build")

USE_PYBIND11 = False

if USE_PYBIND11:
    import fabular_py11 as fbl
else:
    import fabular as fbl

def test_torch():
    print("=== Testing PyTorch Tensor ===")
    
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    cuda_tensor = torch.tensor([[1.1, 2.2], [3.3, 4.4]], device="cuda")
    int_tensor = torch.randint(0, 10, (3,), dtype=torch.int64)
    
    dl_cpu = cpu_tensor.__dlpack__()
    dl_cuda = cuda_tensor.__dlpack__()
    dl_int = int_tensor.__dlpack__()
    
    fbl.print_dlpack(dl_cpu)
    fbl.print_dlpack(dl_cuda)
    fbl.print_dlpack(dl_int)

def test_paddle():
    if not has_paddle:
        print("PaddlePaddle not installed, skipping tests")
        return
        
    print("\n=== Testing PaddlePaddle Tensor ===")
    
    cpu_tensor = paddle.to_tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    cuda_tensor = paddle.to_tensor([[5.5, 4.4], [3.3, 2.2]], place=paddle.CUDAPlace(0))
    int_tensor = paddle.randint(0, 10, (5,), dtype="int64")
    
    dl_cpu = paddle.utils.dlpack.to_dlpack(cpu_tensor)
    dl_cuda = paddle.utils.dlpack.to_dlpack(cuda_tensor)
    dl_int = paddle.utils.dlpack.to_dlpack(int_tensor)
    
    fbl.print_dlpack(dl_cpu)
    fbl.print_dlpack(dl_cuda)
    fbl.print_dlpack(dl_int)

def test_numpy():
    print("\n=== Testing NumPy Array via PyTorch ===")
    
    np_array = np.array([10, 20, 30, 40], dtype=np.float32)
    
    torch_tensor = torch.from_numpy(np_array)
    dl_np = torch_tensor.__dlpack__()
    
    fbl.print_dlpack(dl_np)

if __name__ == "__main__":
    print("Testing torch")
    test_torch()
    
    if has_paddle:
        print("Testing paddle")
        test_paddle()
    else:
        print("\nSkipping PaddlePaddle tests")
    
    print("Testing numpy")
    test_numpy()