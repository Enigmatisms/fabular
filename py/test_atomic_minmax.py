import sys
sys.path.append("../build")

import fabular as fbl
import numpy as np
import torch
from test_utils import try_test

cnt = 0

def make_case_and_test(input_shape, dtype, dtype_index, op = 'min'):
    global cnt
    print(f"{dtype} Case: {cnt + 1}")
    cnt += 1

    init_val = 32767 if op == 'min' else -32767
    inout_tensor = torch.full(input_shape[:-1], init_val, dtype=dtype).cuda()

    to_assign_tensor = torch.randint(-32767, 32767, input_shape[:-1], dtype=dtype).cuda()
    gt_tensor = to_assign_tensor.clone()
    to_assign_tensor = to_assign_tensor.unsqueeze(-1).expand(input_shape).contiguous()
    
    dl_src = to_assign_tensor.__dlpack__()
    dl_dst = inout_tensor.__dlpack__()
    if op == 'min':
        fbl.min_last_dim(dl_src, dl_dst)
    else:
        fbl.max_last_dim(dl_src, dl_dst)

    print(to_assign_tensor.squeeze())
    try_test(inout_tensor.cpu().numpy(), gt_tensor.cpu().numpy(), "Fabular", 'atomic min/max')

if __name__ == "__main__":
    torch.manual_seed(114514)

    for i, dtype in enumerate([torch.int16]):
        make_case_and_test([63, 129, 266], dtype, i)

    
