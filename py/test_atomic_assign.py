import sys
sys.path.append("../build")

import fabular as fbl
import numpy as np
import torch
import paddle
from test_utils import try_test

cnt = 0

def make_case_and_test(input_shape, dtype, value_to_set = 1, total_test_per_dtype = 7):
    global cnt
    print(f"{dtype} Case: {(cnt % total_test_per_dtype) + 1}")
    cnt += 1

    inout_tensor = torch.zeros(*input_shape[:-1], dtype=dtype).cuda()
    
    to_assign_tensor = torch.full(input_shape, value_to_set, dtype=dtype).cuda()
    gt_tensor = torch.full_like(inout_tensor, value_to_set)

    dl_src = to_assign_tensor.__dlpack__()
    dl_dst = inout_tensor.__dlpack__()
    fbl.atomic_assign(dl_src, dl_dst)

    try_test(inout_tensor.cpu().numpy(), gt_tensor.cpu().numpy(), "Fabular")
    

if __name__ == "__main__":
    torch.manual_seed(114514)

    for dtype in [torch.float32, torch.int32, torch.int64, torch.float64, torch.float16, torch.uint8, torch.int16]:
        make_case_and_test([3, 4, 5, 8], dtype)
        make_case_and_test([3, 32, 10, 17], dtype)
        make_case_and_test([128, 4], dtype)
        make_case_and_test([3, 2], dtype)
        make_case_and_test([16, 32], dtype)
        make_case_and_test([2, 2, 2, 4, 4], dtype)
        make_case_and_test([37, 1], dtype)
