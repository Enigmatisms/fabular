import numpy as np

def try_test(actual_np: np.ndarray, gt_np: np.ndarray, actual_name: str, op: str = 'put_along_axis'):
    try:
        np.testing.assert_allclose(actual_np, gt_np)
    except AssertionError as ex_msg:
        print(f"{actual_name} {op} differs from PyTorch!")
        print(f"{actual_name} (dtype = {actual_np.dtype}, shape = {actual_np.shape}):")
        print(actual_np)
        print(f"PyTorch (dtype = {gt_np.dtype}, shape = {gt_np.shape}):")
        print(gt_np)
        print(ex_msg)