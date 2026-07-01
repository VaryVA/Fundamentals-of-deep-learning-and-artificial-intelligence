import torch
import time


device = "cuda" if torch.cuda.is_available() else "cpu"


def create_data():
    return [
        torch.rand(64, 1024, 1024),
        torch.rand(128, 512, 512),
        torch.rand(256, 256, 256),
    ]


def measure_cpu(operation):
    start = time.time()
    operation()
    end = time.time()
    return (end - start) * 1000


def measure_gpu(operation):
    if not torch.cuda.is_available():
        return None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    operation()

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)


def benchmark_operation(name, cpu_op, gpu_op):
    cpu_time = measure_cpu(cpu_op)
    gpu_time = measure_gpu(gpu_op)

    if gpu_time is not None:
        speedup = cpu_time / gpu_time
        print(f"{name:20} | {cpu_time:8.2f} | {gpu_time:8.2f} | {speedup:.2f}x")
    else:
        print(f"{name:20} | {cpu_time:8.2f} | GPU unavailable")


def main():
    data = create_data()
    A = data[0]
    B = data[0]

    A_gpu = A.to(device)
    B_gpu = B.to(device)

    print("Operation            | CPU(ms)  | GPU(ms)  | Ускорение")

    benchmark_operation(
        "Matmul",
        lambda: torch.matmul(A, B.transpose(-1, -2)),
        lambda: torch.matmul(A_gpu, B_gpu.transpose(-1, -2))
    )

    benchmark_operation(
        "Addition",
        lambda: A + B,
        lambda: A_gpu + B_gpu
    )

    benchmark_operation(
        "Multiplication",
        lambda: A * B,
        lambda: A_gpu * B_gpu
    )

    benchmark_operation(
        "Transpose",
        lambda: A.transpose(-1, -2),
        lambda: A_gpu.transpose(-1, -2)
    )

    benchmark_operation(
        "Sum",
        lambda: A.sum(),
        lambda: A_gpu.sum()
    )


if __name__ == "__main__":
    main()