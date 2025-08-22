import sys
sys.path.append("../build")

import fabular as fbl
import numpy as np
import torch
import paddle
from test_utils import try_test

def make_case_and_test(input_shape, index_shape, dim, indices_range, value_to_set = 6):
    inout_tensor = torch.zeros(*input_shape, dtype=torch.float32).cuda()
    gt_tensor = inout_tensor.clone()
    indices = torch.randint(0, indices_range, index_shape, dtype=torch.int64).cuda()

    result = torch.scatter(gt_tensor, dim, indices, value_to_set)
    pd_input = paddle.zeros(*input_shape, dtype=paddle.float32)
    pd_indices = paddle.to_tensor(indices.cpu().numpy())
    pd_result = paddle.put_along_axis(pd_input, indices = pd_indices, values = value_to_set, axis = dim, broadcast = False)

    dl_inout = inout_tensor.__dlpack__()
    dl_indices = indices.__dlpack__()
    fbl.put_along_axis(dl_inout, dl_indices, dim)

    try_test(inout_tensor.cpu().numpy(), result.cpu().numpy(), "Fabular")
    try_test(pd_result.numpy(), result.cpu().numpy(), "Paddle")
    

if __name__ == "__main__":
    torch.manual_seed(114514)

    print("Case 1:")
    make_case_and_test((2, 3, 4, 5), (2, 2, 4, 5), dim = 2, indices_range = 3)
    print("Case 2:")
    make_case_and_test((2, 3, 4, 5), (2, 1, 4, 5), dim = 3, indices_range = 5)
    print("Case 3:")
    make_case_and_test((2, 2, 3, 2, 5), (2, 2, 2, 2, 5), dim = 3, indices_range = 2)
    print("Case 4:")
    make_case_and_test((2, 5, 2, 3, 2), (2, 5, 2, 2, 2), dim = 1, indices_range = 5)

    print("\nExpect same cases:\n")
    print("Case 5:")
    make_case_and_test((8, 8, 8), (7, 8, 8), dim = 1, indices_range = 8)

    print("Case 6:")
    make_case_and_test((257, 63, 128), (257, 63, 128), dim = 0, indices_range = 128)

    print("Case 7:")
    make_case_and_test((37, 18), (31, 16), dim = 0, indices_range = 20)

    print("Case 8:")
    make_case_and_test((129, 2), (16, 2), dim = 0, indices_range = 64)

    print("Case 9:")
    make_case_and_test((1023,), (1023,), dim = 0, indices_range = 512)

    print("Case 10:")
    make_case_and_test((1023,), (167,), dim = 0, indices_range = 512)
    