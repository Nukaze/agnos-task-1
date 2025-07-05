#!/usr/bin/env python3
"""
Diagnostic script to check CUDA setup and PyTorch installation
"""

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    return sys.version_info >= (3, 8)

def check_pytorch_installation():
    """Check PyTorch installation"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch build: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'CPU only'}")
        return True
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def check_cuda_availability():
    """Check CUDA availability"""
    try:
        import torch
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            
            # Test CUDA functionality
            try:
                test_tensor = torch.tensor([1.0], device="cuda")
                print("✅ CUDA test passed - tensor created successfully")
                return True
            except Exception as e:
                print(f"❌ CUDA test failed: {e}")
                return False
        else:
            print("❌ CUDA is not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed")
            print("NVIDIA-SMI output:")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA driver not found or not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA driver may not be installed")
        return False

def check_sentence_transformers():
    """Check sentence-transformers installation"""
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers is installed")
        
        # Test model loading
        try:
            model = SentenceTransformer('BAAI/bge-m3')
            print("✅ BGE-M3 model can be loaded")
            return True
        except Exception as e:
            print(f"❌ Error loading BGE-M3 model: {e}")
            return False
            
    except ImportError:
        print("❌ sentence-transformers is not installed")
        return False

def main():
    print("=== CUDA and PyTorch Diagnostic ===")
    print()
    
    # Check Python
    print("1. Python Version:")
    check_python_version()
    print()
    
    # Check NVIDIA driver
    print("2. NVIDIA Driver:")
    nvidia_ok = check_nvidia_driver()
    print()
    
    # Check PyTorch
    print("3. PyTorch Installation:")
    pytorch_ok = check_pytorch_installation()
    print()
    
    # Check CUDA
    print("4. CUDA Availability:")
    cuda_ok = check_cuda_availability()
    print()
    
    # Check sentence-transformers
    print("5. Sentence Transformers:")
    st_ok = check_sentence_transformers()
    print()
    
    # Summary
    print("=== SUMMARY ===")
    if all([pytorch_ok, cuda_ok, st_ok]):
        print("✅ All components are working correctly!")
        print("Your RAG service should work with CUDA acceleration.")
    else:
        print("❌ Some components have issues:")
        if not pytorch_ok:
            print("  - Install PyTorch: pip install torch torchvision")
        if not cuda_ok:
            print("  - Install CUDA version of PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        if not st_ok:
            print("  - Install sentence-transformers: pip install sentence-transformers")
    
    print()
    print("=== RECOMMENDATIONS ===")
    if not nvidia_ok:
        print("1. Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
    if not cuda_ok and nvidia_ok:
        print("2. Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
    if not pytorch_ok:
        print("3. Install PyTorch with CUDA support")
    print("4. Restart your application after making changes")

if __name__ == "__main__":
    main() 