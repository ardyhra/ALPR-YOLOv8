import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Bagian ini hanya akan berjalan jika CUDA tersedia
if torch.cuda.is_available():
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    # Cek build PyTorch
    build_version = torch.__version__.split('+')[-1]
    if build_version.startswith('cu'):
        print(f"PyTorch build is for CUDA: {build_version}")
        print("Namun, PyTorch tidak dapat menemukan instalasi CUDA yang kompatibel di sistem Anda.")
        print("Pastikan NVIDIA Driver dan CUDA Toolkit versi yang sesuai sudah terinstall dengan benar.")
    else:
        print("PyTorch build ini adalah versi CPU-only.")

