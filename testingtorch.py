import torch

# Check if CUDA is available
print("Is CUDA available?", torch.cuda.is_available())

# Check CUDA version
print("CUDA version:", torch.version.cuda)

# Check the current device
if torch.cuda.is_available():
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Run a simple tensor operation on the GPU
    tensor = torch.rand(3, 3).cuda()
    print("Tensor on GPU:", tensor)
else:
    print("CUDA is not available.")
