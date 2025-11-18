import sys
import tqdm
sys.path.append("../build")

import fabular as fbl
import torch

def testing(length):
    arr_len = length
    input_tensor = torch.arange(arr_len, dtype=torch.int32).cuda()
    output_tensor = torch.zeros([arr_len * 4], dtype=torch.int32).cuda()

    dl_in = input_tensor.__dlpack__()
    dl_out = output_tensor.__dlpack__()
    fbl.unordered_expand(dl_in, dl_out)

    result = output_tensor.reshape([-1, 4])
    sliced = set(result[:, 0].cpu().tolist())
    assert(len(sliced) == arr_len)

if __name__ == "__main__":
    test_time = 400
    tests = [32, 216, 277, 512, 593, 756, 2333, 8992, 13211]
    for i in tqdm.tqdm(range(test_time)):
        # print(f"Test {i+1}/{test_time} begins...")
        for test_len in tests:
            testing(test_len)
            # print(f"Tests: length {test_len} done")