import sys
sys.path.append("../build")

import fabular as fbl
import numpy as np
import torch
import paddle
from test_utils import try_test

cnt = 0

def make_case_and_test(input_shape, dtype, dtype_index, value_to_set = 1, total_test_per_dtype = 12):
    global cnt
    print(f"{dtype} Case: {(cnt % total_test_per_dtype) + 1}")
    cnt += 1

    inout_tensor = torch.zeros(*input_shape[:-1], dtype=dtype).cuda()

    if dtype_index < 4:
        to_assign_tensor = torch.randint(0, 256, input_shape[:-1], dtype=dtype).cuda()
    else:
        to_assign_tensor = torch.rand(input_shape[:-1], dtype=dtype).cuda()
    gt_tensor = to_assign_tensor.clone()
    to_assign_tensor = to_assign_tensor.unsqueeze(-1).expand(input_shape).contiguous()
    
    dl_src = to_assign_tensor.__dlpack__()
    dl_dst = inout_tensor.__dlpack__()
    fbl.atomic_assign(dl_src, dl_dst)

    try_test(inout_tensor.cpu().numpy(), gt_tensor.cpu().numpy(), "Fabular")

def make_no_contention_case_and_test(input_shape, dtype, dtype_index, total_test_per_dtype = 12):
    global cnt
    print(f"{dtype} Case: {(cnt % total_test_per_dtype) + 1}")
    cnt += 1
    if input_shape[-1] != 1:
        input_shape.append(1)

    inout_tensor = torch.zeros(*input_shape[:-1], dtype=dtype).cuda()
    
    if dtype_index < 4:
        to_assign_tensor = torch.randint(0, 256, input_shape, dtype=dtype).cuda()
    else:
        to_assign_tensor = torch.rand(input_shape, dtype=dtype).cuda()
    gt_tensor = to_assign_tensor.squeeze(-1).clone()

    dl_src = to_assign_tensor.__dlpack__()
    dl_dst = inout_tensor.__dlpack__()
    fbl.atomic_assign(dl_src, dl_dst)

    try_test(inout_tensor.cpu().numpy(), gt_tensor.cpu().numpy(), "Fabular")

if __name__ == "__main__":
    torch.manual_seed(114514)

    for i, dtype in enumerate([torch.int32, torch.int64, torch.uint8, torch.int16, torch.float64, torch.float16, torch.float32]):
        make_case_and_test([3, 4, 5, 8], dtype, i)
        make_case_and_test([3, 32, 10, 17], dtype, i)
        make_case_and_test([128, 4], dtype, i)
        make_case_and_test([3, 2], dtype, i)
        make_case_and_test([16, 32], dtype, i)
        make_case_and_test([2, 2, 2, 4, 4], dtype, i)
        make_case_and_test([37, 1], dtype, i)
        make_case_and_test([327, 58, 1], dtype, i)
        make_case_and_test([32768, 1], dtype, i)
        make_case_and_test([1, 1], dtype, i)
        make_case_and_test([10, 10, 10, 10], dtype, i)
        make_case_and_test([4, 1], dtype, i)

    print("\nContention test over, non-contention test starts:\n")
    cnt = 0
    for i, dtype in enumerate([torch.int32, torch.int64, torch.uint8, torch.int16, torch.float64, torch.float16, torch.float32]):
        make_no_contention_case_and_test([3, 4, 5, 8], dtype, i)
        make_no_contention_case_and_test([3, 32, 10, 17], dtype, i)
        make_no_contention_case_and_test([128, 4], dtype, i)
        make_no_contention_case_and_test([3, 2], dtype, i)
        make_no_contention_case_and_test([16, 32], dtype, i)
        make_no_contention_case_and_test([2, 2, 2, 4, 4], dtype, i)
        make_no_contention_case_and_test([37, 1], dtype, i)
        make_no_contention_case_and_test([327, 58, 1], dtype, i)
        make_no_contention_case_and_test([32768, 1], dtype, i)
        make_no_contention_case_and_test([1, 1], dtype, i)
        make_no_contention_case_and_test([10, 10, 10, 10], dtype, i)
        make_no_contention_case_and_test([4, 1], dtype, i)
