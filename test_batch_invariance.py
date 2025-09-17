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
    identical results for the first row.
    """
    print("Testing matmul batch invariance across multiple batch sizes...")
    
    # Test parameters - using larger matrices to trigger potential CUDA kernel differences
    max_batch = 2048
    D = 4096
    test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    # Create input matrices with more challenging values
    # Use different random seeds to avoid potential cache effects
    torch.manual_seed(12345)
    full_a = torch.randn(max_batch, D, dtype=torch.float16) * 10.0  # Larger values
    torch.manual_seed(54321)
    b = torch.randn(D, D, dtype=torch.float16) * 5.0
    
    # Force CUDA synchronization to avoid timing effects
    torch.cuda.synchronize()
    
    # Reference: compute first row only
    reference_result = torch.mm(full_a[:1], b)
    torch.cuda.synchronize()
    
    all_passed = True
    max_diff_overall = 0.0
    
    for batch_size in test_batch_sizes:
        # Clear CUDA cache between runs to avoid cache effects
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Compute with current batch size, then extract the first row
        current_result = torch.mm(full_a[:batch_size], b)[:1]
        torch.cuda.synchronize()
        
        # Compare with reference - use more sensitive comparison
        diff = (reference_result - current_result).abs().max().item()
        max_diff_overall = max(max_diff_overall, diff)
        
        # Use a small tolerance instead of exact equality for floating point
        tolerance = 1e-6 if torch.get_default_dtype() == torch.float32 else 1e-3
        passed = diff <= tolerance
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Batch size {batch_size:4d}: {status} (max diff: {diff:.6e})")
        
        if not passed:
            print(f"    Expected batch invariance (diff <= {tolerance:.0e}), but got difference of {diff:.6e}")
    
    print(f"\nOverall result: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print(f"Maximum difference across all batch sizes: {max_diff_overall:.6e}")
    
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
print("\n3. Testing multiple batch sizes with Batch-Invariant Mode:")
with set_batch_invariant_mode(True):
    invariant_passed = test_matmul_batch_invariance_multiple_sizes()


print("\n" + "="*80)
print("FINAL RESULTS:")
print(f"Standard PyTorch multiple batch sizes: {'✓ PASSED' if standard_passed else '✗ FAILED'}")
print(f"Batch-Invariant multiple batch sizes:  {'✓ PASSED' if invariant_passed else '✗ FAILED'}")
print("="*80)

