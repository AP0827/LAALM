#!/usr/bin/env python3
"""
Auto-AVSR Setup Verification Script
This script verifies that all required packages are installed correctly
and checks system capabilities for running auto-avsr.
"""

import sys
from typing import List, Tuple

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Try to import a module and return success status and version
    
    Args:
        module_name: Name of the module to import
        package_name: Display name for the package (if different from module_name)
    
    Returns:
        Tuple of (success: bool, version: str)
    """
    if package_name is None:
        package_name = module_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    """Main verification routine"""
    
    print_header("Auto-AVSR Setup Verification")
    print("Checking installation of all required packages...")
    
    all_passed = True
    results: List[Tuple[str, bool, str]] = []
    
    # Core PyTorch packages
    print_header("Core PyTorch Packages")
    
    core_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
    ]
    
    for module, display_name in core_packages:
        success, version = check_import(module, display_name)
        results.append((display_name, success, version))
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8} {display_name:20} {version}")
        all_passed = all_passed and success
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        status = "✓ PASS" if cuda_available else "⚠ WARN"
        print(f"\n  {status:8} {'CUDA Support':20} {cuda_version}")
        if cuda_available:
            print(f"           {'GPU Devices':20} {cuda_device_count}")
            for i in range(cuda_device_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"           {'  Device ' + str(i):20} {gpu_name}")
        else:
            print("           Note: CUDA not available. Training will use CPU (much slower).")
    except Exception as e:
        print(f"  ✗ FAIL   CUDA Check          {e}")
    
    # Auto-AVSR specific packages
    print_header("Auto-AVSR Core Packages")
    
    avsr_packages = [
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('sentencepiece', 'SentencePiece'),
        ('av', 'PyAV'),
    ]
    
    for module, display_name in avsr_packages:
        success, version = check_import(module, display_name)
        results.append((display_name, success, version))
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8} {display_name:20} {version}")
        all_passed = all_passed and success
    
    # Preprocessing packages
    print_header("Preprocessing Packages")
    
    prep_packages = [
        ('cv2', 'OpenCV'),
        ('ffmpeg', 'FFmpeg-Python'),
        ('skimage', 'Scikit-Image'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('tqdm', 'TQDM'),
    ]
    
    for module, display_name in prep_packages:
        success, version = check_import(module, display_name)
        results.append((display_name, success, version))
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8} {display_name:20} {version}")
        all_passed = all_passed and success
    
    # Face detection packages
    print_header("Face Detection Packages")
    
    face_packages = [
        ('mediapipe', 'MediaPipe'),
    ]
    
    for module, display_name in face_packages:
        success, version = check_import(module, display_name)
        results.append((display_name, success, version))
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8} {display_name:20} {version}")
        all_passed = all_passed and success
    
    # Test basic functionality
    print_header("Functionality Tests")
    
    # Test tensor creation
    try:
        import torch
        tensor = torch.randn(3, 3)
        print(f"  ✓ PASS   Tensor Creation     Shape: {tensor.shape}")
    except Exception as e:
        print(f"  ✗ FAIL   Tensor Creation     {e}")
        all_passed = False
    
    # Test CUDA tensor (if available)
    try:
        import torch
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(3, 3).cuda()
            print(f"  ✓ PASS   CUDA Tensor          Shape: {cuda_tensor.shape}")
        else:
            print(f"  ⚠ SKIP   CUDA Tensor          (CUDA not available)")
    except Exception as e:
        print(f"  ✗ FAIL   CUDA Tensor          {e}")
    
    # Test SentencePiece model path
    try:
        import os
        spm_model_path = "./spm/unigram/unigram5000.model"
        spm_vocab_path = "./spm/unigram/unigram5000_units.txt"
        
        if os.path.exists(spm_model_path):
            print(f"  ✓ PASS   SentencePiece Model Found at {spm_model_path}")
        else:
            print(f"  ⚠ WARN   SentencePiece Model Not found at {spm_model_path}")
            print(f"           (Required for training/inference on English datasets)")
    except Exception as e:
        print(f"  ✗ FAIL   SentencePiece Check {e}")
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n  Total Packages: {total}")
    print(f"  Passed:         {passed}")
    print(f"  Failed:         {total - passed}")
    
    if all_passed:
        print("\n  🎉 SUCCESS! All packages are installed correctly.")
        print("  You're ready to use Auto-AVSR!")
        print("\n  Next steps:")
        print("    1. Prepare your dataset (see preparation/README.md)")
        print("    2. Train a model: python train.py --help")
        print("    3. Evaluate a model: python eval.py --help")
        return 0
    else:
        print("\n  ❌ FAILED! Some packages are missing or not installed correctly.")
        print("  Please install missing packages before using Auto-AVSR.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
