import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - reserved_memory

    print(f"Total GPU memory: {total_memory / (1024**3):.2f} GB")
    print(f"Reserved GPU memory: {reserved_memory / (1024**3):.2f} GB")
    print(f"Allocated GPU memory: {allocated_memory / (1024**3):.2f} GB")
    print(f"Available GPU memory: {available_memory / (1024**3):.2f} GB")
else:
    print("CUDA is not available.")
