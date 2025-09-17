import torch
from batch_invariant_ops import set_batch_invariant_mode
torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)
    
    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]
    
    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0


def test_matmul_batch_invariance_multiple_sizes():
    """
    Test that matmul produces batch-invariant results across different batch sizes.
    This test verifies that computing matmul with different batch sizes yields
    identical results for the same input rows.
    """
    print("Testing matmul batch invariance across multiple batch sizes...")
    
    # Test parameters
    max_batch = 1024
    D = 2048
    test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Create input matrices
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    full_a = torch.randn(max_batch, D, dtype=torch.float16)
    b = torch.randn(D, D, dtype=torch.float16)
    
    # Reference: compute with batch size 1
    reference_result = torch.mm(full_a[:1], b)
    
    all_passed = True
    max_diff_overall = 0.0
    
    for batch_size in test_batch_sizes:
        # Compute with current batch size, then take first row
        current_result = torch.mm(full_a[:batch_size], b)[:1]
        
        # Compare with reference
        diff = (reference_result - current_result).abs().max().item()
        max_diff_overall = max(max_diff_overall, diff)
        
        passed = diff == 0.0
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Batch size {batch_size:4d}: {status} (max diff: {diff:.2e})")
        
        if not passed:
            print(f"    Expected batch invariance, but got difference of {diff}")
    
    print(f"\nOverall result: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print(f"Maximum difference across all batch sizes: {max_diff_overall:.2e}")
    
    return all_passed


def test_matmul_batch_invariance_different_dtypes():
    """
    Test batch invariance across different data types.
    """
    print("Testing matmul batch invariance across different dtypes...")
    
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    batch_sizes = [1, 16, 128]
    M, N, K = 512, 512, 512
    
    all_passed = True
    
    for dtype in dtypes:
        print(f"\n  Testing dtype: {dtype}")
        torch.manual_seed(42)  # Fixed seed for reproducibility
        
        # Create test matrices
        a = torch.randn(max(batch_sizes), K, dtype=dtype)
        b = torch.randn(K, N, dtype=dtype)
        
        # Reference with batch size 1
        ref_result = torch.mm(a[:1], b)
        
        dtype_passed = True
        for batch_size in batch_sizes[1:]:  # Skip batch_size=1 as it's the reference
            current_result = torch.mm(a[:batch_size], b)[:1]
            diff = (ref_result - current_result).abs().max().item()
            
            passed = diff == 0.0
            dtype_passed = dtype_passed and passed
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"    Batch size {batch_size:3d}: {status} (max diff: {diff:.2e})")
        
        all_passed = all_passed and dtype_passed
        print(f"    {dtype}: {'✓ PASSED' if dtype_passed else '✗ FAILED'}")
    
    return all_passed

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

print("\n" + "="*80)
print("COMPREHENSIVE BATCH INVARIANCE TESTS")
print("="*80)

# Test multiple batch sizes with standard PyTorch
print("\n1. Testing multiple batch sizes with Standard PyTorch:")
with set_batch_invariant_mode(False):
    standard_passed = test_matmul_batch_invariance_multiple_sizes()

# Test multiple batch sizes with batch-invariant mode
print("\n2. Testing multiple batch sizes with Batch-Invariant Mode:")
with set_batch_invariant_mode(True):
    invariant_passed = test_matmul_batch_invariance_multiple_sizes()

# Test different data types with batch-invariant mode
print("\n3. Testing different dtypes with Batch-Invariant Mode:")
with set_batch_invariant_mode(True):
    dtype_passed = test_matmul_batch_invariance_different_dtypes()

print("\n" + "="*80)
print("FINAL RESULTS:")
print(f"Standard PyTorch multiple batch sizes: {'✓ PASSED' if standard_passed else '✗ FAILED'}")
print(f"Batch-Invariant multiple batch sizes:  {'✓ PASSED' if invariant_passed else '✗ FAILED'}")
print(f"Batch-Invariant different dtypes:      {'✓ PASSED' if dtype_passed else '✗ FAILED'}")
print("="*80)

"""
Standard PyTorch:
Difference: 10.7294921875
Deterministic: False

Batch-Invariant Mode:
Difference: 0.0
Deterministic: True

================================================================================
COMPREHENSIVE BATCH INVARIANCE TESTS
================================================================================

1. Testing multiple batch sizes with Standard PyTorch:
Testing matmul batch invariance across multiple batch sizes...
  Batch size    1: ✓ PASS (max diff: 0.00e+00)
  Batch size    2: ✓ PASS (max diff: 0.00e+00)
  Batch size    4: ✓ PASS (max diff: 0.00e+00)
  Batch size    8: ✓ PASS (max diff: 0.00e+00)
  Batch size   16: ✓ PASS (max diff: 0.00e+00)
  Batch size   32: ✓ PASS (max diff: 0.00e+00)
  Batch size   64: ✓ PASS (max diff: 0.00e+00)
  Batch size  128: ✓ PASS (max diff: 0.00e+00)
  Batch size  256: ✓ PASS (max diff: 0.00e+00)
  Batch size  512: ✓ PASS (max diff: 0.00e+00)
  Batch size 1024: ✓ PASS (max diff: 0.00e+00)

Overall result: ✓ ALL PASSED
Maximum difference across all batch sizes: 0.00e+00

2. Testing multiple batch sizes with Batch-Invariant Mode:
Testing matmul batch invariance across multiple batch sizes...
  Batch size    1: ✓ PASS (max diff: 0.00e+00)
  Batch size    2: ✓ PASS (max diff: 0.00e+00)
  Batch size    4: ✓ PASS (max diff: 0.00e+00)
  Batch size    8: ✓ PASS (max diff: 0.00e+00)
  Batch size   16: ✓ PASS (max diff: 0.00e+00)
  Batch size   32: ✓ PASS (max diff: 0.00e+00)
  Batch size   64: ✓ PASS (max diff: 0.00e+00)
  Batch size  128: ✓ PASS (max diff: 0.00e+00)
  Batch size  256: ✓ PASS (max diff: 0.00e+00)
  Batch size  512: ✓ PASS (max diff: 0.00e+00)
  Batch size 1024: ✓ PASS (max diff: 0.00e+00)

Overall result: ✓ ALL PASSED
Maximum difference across all batch sizes: 0.00e+00

3. Testing different dtypes with Batch-Invariant Mode:
Testing matmul batch invariance across different dtypes...

  Testing dtype: torch.float16
    Batch size  16: ✓ PASS (max diff: 0.00e+00)
    Batch size 128: ✓ PASS (max diff: 0.00e+00)
    torch.float16: ✓ PASSED

  Testing dtype: torch.bfloat16
    Batch size  16: ✓ PASS (max diff: 0.00e+00)
    Batch size 128: ✓ PASS (max diff: 0.00e+00)
    torch.bfloat16: ✓ PASSED

  Testing dtype: torch.float32
    Batch size  16: ✓ PASS (max diff: 0.00e+00)
    Batch size 128: ✓ PASS (max diff: 0.00e+00)
    torch.float32: ✓ PASSED

================================================================================
FINAL RESULTS:
Standard PyTorch multiple batch sizes: ✓ PASSED
Batch-Invariant multiple batch sizes:  ✓ PASSED
Batch-Invariant different dtypes:      ✓ PASSED
================================================================================
"""