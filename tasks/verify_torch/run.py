import torch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(4, 4, device=device)
    y = x @ x.T

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"tensor_sum: {y.sum().item():.6f}")


if __name__ == "__main__":
    main()
