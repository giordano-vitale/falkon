import torch
import os
import socket

def sanity_check():
    hostname = socket.gethostname()
    print(f"\n--- Report from Node: {hostname} ---")
    
    # Check if CUDA is even available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("RESULT: FAILURE - No GPUs detected by PyTorch.")
        return

    # Count GPUs
    device_count = torch.cuda.device_count()
    print(f"GPU Count detected: {device_count}")
    
    # Check Slurm Environment Variables
    job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
    proc_id = os.environ.get('SLURM_PROCID', 'N/A')
    print(f"Slurm Job ID: {job_id} | Slurm Process ID: {proc_id}")

    # Test each GPU
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {props.name} ({props.total_memory / 1024**2:.0f} MB)")
        
        # Perform a small dummy calculation
        try:
            x = torch.tensor([1.0, 2.0], device=f"cuda:{i}")
            y = x * 2
            print(f"    Calc test: SUCCESS")
        except Exception as e:
            print(f"    Calc test: FAILED with error: {e}")

if __name__ == "__main__":
    sanity_check()