import sys
sys.path.append("../build")

import fabular as fbl
import numpy as np
import torch
import paddle

if __name__ == "__main__":
    inout_tensor = torch.zeros(2, 3, 4, 5).cuda()
    gt_tensor = inout_tensor.clone()
    indices = torch.randint(0, 3, (2, 2, 4, 5), dtype=torch.int64).cuda()
    dim = 2

    result = torch.scatter(gt_tensor, dim, indices, 6)
    print("Result gt:\n", result.cpu().numpy())

    dl_inout = inout_tensor.__dlpack__()
    dl_indices = indices.__dlpack__()
    fbl.put_along_axis(dl_inout, dl_indices, dim)
    print("Result mine:\n", inout_tensor.cpu().numpy())
    np.testing.assert_allclose(inout_tensor.cpu().numpy(), result.cpu().numpy())