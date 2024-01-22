import torch

if torch.cuda.is_available():
    
    num_gpus = torch.cuda.device_count()
    
    print(f"CUDA is available with {num_gpus} GPU(s).")
    
    
    current_gpu_name = torch.cuda.get_device_name(0) 
    print(f"Current GPU: {current_gpu_name}")
else:
    print("CUDA is not available. Running on CPU.")
