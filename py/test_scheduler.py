import sys
sys.path.append("../build")

import fabular as fbl
import torch

if __name__ == "__main__":
    arr_len = 256
    input_tensor = torch.arange(arr_len, dtype=torch.int32).cuda()
    output_tensor = torch.zeros([arr_len * 4], dtype=torch.int32).cuda()

    dl_in = input_tensor.__dlpack__()
    dl_out = output_tensor.__dlpack__()
    print("Function starts...")
    fbl.unordered_expand(dl_in, dl_out)
    print("Function returned")

    result = output_tensor.reshape([-1, 4])
    sliced = set(result[:, 0].cpu().tolist())
    assert(len(sliced) == arr_len)
