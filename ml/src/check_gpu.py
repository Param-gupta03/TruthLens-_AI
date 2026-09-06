"""
check_gpu.py - GPU and CUDA Environment Diagnostic for TruthLens AI.

Verifies:
- NVIDIA GPU detection
- CUDA availability in PyTorch
- GPU name, architecture, and memory
- CUDA and cuDNN versions
- Tensor execution on GPU (small CUDA tensor test)
- Sets device = cuda or raises RuntimeError if unavailable.
"""

import os
import sys
import subprocess
import torch


def check_system_nvidia_smi() -> dict:
    """Queries nvidia-smi for system-level driver and GPU information."""
    info = {"driver_available": False, "gpu_name": "Unknown", "driver_version": "Unknown", "cuda_driver_version": "Unknown"}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = res.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                info["driver_available"] = True
                info["gpu_name"] = parts[0]
                info["driver_version"] = parts[1]
                info["total_vram_mb"] = float(parts[2])
    except Exception:
        pass
    return info


def inspect_gpu_environment(require_cuda: bool = True) -> dict:
    """
    Inspects and validates the PyTorch CUDA runtime environment.
    Raises RuntimeError if require_cuda is True and CUDA is unavailable.
    """
    print("=" * 70)
    print(" TRUTHLENS AI — HARDWARE & ACCELERATION DIAGNOSTIC")
    print("=" * 70)
    
    sys_gpu = check_system_nvidia_smi()
    if sys_gpu.get("driver_available"):
        print(f"System GPU (nvidia-smi) : {sys_gpu.get('gpu_name')}")
        print(f"NVIDIA Driver Version   : {sys_gpu.get('driver_version')}")
        print(f"Total VRAM (Hardware)   : {sys_gpu.get('total_vram_mb', 0):,.0f} MB ({sys_gpu.get('total_vram_mb', 0)/1024:.2f} GB)")
    else:
        print("System GPU (nvidia-smi) : Not detected or nvidia-smi unavailable")
        
    print("-" * 70)
    print(f"PyTorch Version         : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available (PyTorch): {cuda_avail}")
    
    if not cuda_avail:
        err_msg = (
            "\n[CRITICAL ERROR] CUDA is NOT available in PyTorch!\n"
            "An NVIDIA GPU was detected on the system, but PyTorch cannot access CUDA.\n"
            "Training cannot proceed on GPU until PyTorch with CUDA support is active.\n"
            "To resolve: install torch with CUDA wheels (e.g., pip install torch --index-url https://download.pytorch.org/whl/cu124)\n"
        )
        print(err_msg)
        if require_cuda:
            raise RuntimeError("CUDA is required for TruthLens AI training per user configuration.")
        return {"device": "cpu", "cuda_available": False}
        
    # CUDA is available
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    cudnn_avail = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_avail else "N/A"
    
    # Memory metrics
    mem_props = torch.cuda.get_device_properties(0)
    total_mem_gb = mem_props.total_memory / (1024 ** 3)
    compute_cap = torch.cuda.get_device_capability(0)
    
    print(f"GPU Available           : True")
    print(f"GPU Name                : {device_name}")
    print(f"GPU Count               : {device_count}")
    print(f"GPU Compute Capability  : {compute_cap[0]}.{compute_cap[1]}")
    print(f"CUDA Version            : {cuda_version}")
    print(f"cuDNN Available         : {cudnn_avail} (Version {cudnn_version})")
    print(f"GPU Memory (PyTorch)    : {total_mem_gb:.2f} GB")
    print(f"device                  = cuda")
    
    # Run test tensor on GPU
    print("-" * 70)
    try:
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        test_result = (test_tensor * 2).cpu().tolist()
        print(f"[VERIFIED] PyTorch CUDA Tensor Execution: SUCCESS ({test_result})")
    except Exception as e:
        raise RuntimeError(f"Failed to execute test tensor on CUDA device: {e}")
        
    print("=" * 70)
    
    return {
        "device": "cuda",
        "cuda_available": True,
        "gpu_name": device_name,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "total_vram_gb": round(total_mem_gb, 2),
        "compute_capability": compute_cap,
    }


if __name__ == "__main__":
    try:
        inspect_gpu_environment(require_cuda=True)
    except Exception as exc:
        print(f"Inspection halted: {exc}")
        sys.exit(1)
