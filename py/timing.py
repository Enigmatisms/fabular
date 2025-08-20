import torch
import timeit

def test_time(module_name = "fabular"):
    num_test = 50
    exclude_first = 1
    exclude_last = 1
    repeat = 20000 + exclude_first + exclude_last
    execution_times = timeit.repeat(
        "a = fbl.process_dlpack(dl_cuda)",
        setup=f"import torch; import {module_name} as fbl; cuda_tensor = torch.tensor([[1.1, 2.2], [3.3, 4.4]], device='cuda'); dl_cuda = cuda_tensor.__dlpack__()",
        number=num_test,
        repeat=repeat
    )

    total_time = sum(execution_times[exclude_first:-exclude_last])
    actual_repeat = repeat - exclude_first - exclude_last
    avg_time = total_time / actual_repeat
    total_tests = num_test * actual_repeat
    print(f"{module_name} execution time for total of {total_tests} tests: {total_time:.6f} s")
    print(f"Average test time: {(total_time / total_tests * 1e6):.6f} us\n")

if __name__ == "__main__":
    test_time('fabular')
    test_time('fabular_py11')
