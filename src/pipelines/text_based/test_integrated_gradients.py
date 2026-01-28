# src/pipelines/text_based/test_integrated_gradients.py

"""
Test script for Integrated Gradients implementation.
Validates core functionality on a small example.

Usage:
    python -m src.pipelines.text_based.test_integrated_gradients
"""

import torch
import numpy as np
from src.pipelines.text_based.integrated_gradients import (
    CaptumModelWrapper,
    create_baseline,
    compute_integrated_gradients,
    aggregate_attributions_to_events,
    TokenAttributions
)


def test_baseline_creation():
    """Test baseline creation strategies."""
    print("\n" + "="*60)
    print("TEST 1: Baseline Creation")
    print("="*60)
    
    # Mock tokenizer
    class MockTokenizer:
        pad_token_id = 0
        unk_token_id = 1
        mask_token_id = 2
    
    tokenizer = MockTokenizer()
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    
    # Test PAD baseline
    baseline_pad = create_baseline(input_ids, tokenizer, 'pad')
    assert baseline_pad.shape == input_ids.shape, "Baseline shape mismatch"
    assert torch.all(baseline_pad == 0), "PAD baseline should be all 0s"
    print("  ✓ PAD baseline: PASS")
    
    # Test UNK baseline
    baseline_unk = create_baseline(input_ids, tokenizer, 'unk')
    assert torch.all(baseline_unk == 1), "UNK baseline should be all 1s"
    print("  ✓ UNK baseline: PASS")
    
    # Test zero baseline
    baseline_zero = create_baseline(input_ids, tokenizer, 'zero')
    assert torch.all(baseline_zero == 0), "Zero baseline should be all 0s"
    print("  ✓ Zero baseline: PASS")
    
    print("\n  ✅ All baseline tests passed!")


def test_event_aggregation():
    """Test aggregation of token attributions to events."""
    print("\n" + "="*60)
    print("TEST 2: Event Aggregation")
    print("="*60)
    
    # Mock token attributions
    tokens = ["Patient", " visited", " clinic", ";", " Lab", " test", " result", ";", " Diagnosis"]
    attributions = np.array([0.1, 0.2, 0.3, 0.0, 0.5, 0.4, 0.6, 0.0, 0.9])
    
    token_attrs = TokenAttributions(
        attributions=attributions,
        tokens=tokens,
        token_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        baseline_prediction=0.01,
        input_prediction=0.95,
        completeness_score=0.05,
        text="Patient visited clinic; Lab test result; Diagnosis"
    )
    
    # Aggregate by semicolon
    event_attrs = aggregate_attributions_to_events(token_attrs, delimiter=';', aggregation_method='sum')
    
    print(f"  Found {len(event_attrs.event_attributions)} events")
    for i, (event_text, score) in enumerate(event_attrs.event_attributions, 1):
        print(f"    Event {i}: '{event_text[:30]}' → {score:.3f}")
    
    assert len(event_attrs.event_attributions) == 3, "Should have 3 events"
    print("\n  ✅ Event aggregation test passed!")


def test_captum_installation():
    """Test if Captum is installed."""
    print("\n" + "="*60)
    print("TEST 3: Captum Installation")
    print("="*60)
    
    try:
        import captum
        from captum.attr import IntegratedGradients
        print(f"  ✓ Captum version: {captum.__version__}")
        print("\n  ✅ Captum is installed!")
        return True
    except ImportError:
        print("  ✗ Captum is NOT installed!")
        print("  Install with: pip install captum")
        return False


def test_model_wrapper():
    """Test the CaptumModelWrapper."""
    print("\n" + "="*60)
    print("TEST 4: Model Wrapper")
    print("="*60)
    
    # Create a mock model
    class MockClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input_ids, attention_mask):
            batch_size = input_ids.shape[0]
            # Return mock logits
            logits = torch.tensor([[0.2, 0.8]] * batch_size)  # [batch, 2]
            return {'logits': logits}
    
    mock_model = MockClassifier()
    attention_mask = torch.ones((1, 10))
    
    # Wrap model
    wrapped = CaptumModelWrapper(mock_model, attention_mask)
    
    # Test forward
    input_ids = torch.randint(0, 100, (1, 10))
    output = wrapped(input_ids)
    
    assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"
    print(f"  ✓ Wrapper output shape: {output.shape}")
    print(f"  ✓ Cancer logit: {output[0].item():.4f}")
    print("\n  ✅ Model wrapper test passed!")


def test_attribution_properties():
    """Test properties of token attributions."""
    print("\n" + "="*60)
    print("TEST 5: Attribution Properties")
    print("="*60)
    
    # Create mock attributions
    attributions = np.random.randn(100) * 0.5
    tokens = [f"token_{i}" for i in range(100)]
    
    token_attrs = TokenAttributions(
        attributions=attributions,
        tokens=tokens,
        token_ids=np.arange(100),
        baseline_prediction=0.05,
        input_prediction=0.85,
        completeness_score=0.03,
        text="Mock patient record"
    )
    
    # Check properties
    print(f"  Sequence length: {len(token_attrs.tokens)}")
    print(f"  Attribution mean: {np.mean(token_attrs.attributions):.6f}")
    print(f"  Attribution std: {np.std(token_attrs.attributions):.6f}")
    print(f"  Prediction change: {token_attrs.input_prediction - token_attrs.baseline_prediction:.4f}")
    print(f"  Completeness: {token_attrs.completeness_score:.4f}")
    
    # Find top positive and negative
    sorted_indices = np.argsort(token_attrs.attributions)
    print(f"\n  Top positive attribution: {token_attrs.attributions[sorted_indices[-1]]:.4f}")
    print(f"  Top negative attribution: {token_attrs.attributions[sorted_indices[0]]:.4f}")
    
    print("\n  ✅ Attribution properties test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("INTEGRATED GRADIENTS - TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Baseline creation
    try:
        test_baseline_creation()
    except Exception as e:
        print(f"\n  ❌ Baseline test FAILED: {e}")
        all_passed = False
    
    # Test 2: Event aggregation
    try:
        test_event_aggregation()
    except Exception as e:
        print(f"\n  ❌ Event aggregation test FAILED: {e}")
        all_passed = False
    
    # Test 3: Captum installation
    try:
        captum_ok = test_captum_installation()
        if not captum_ok:
            all_passed = False
    except Exception as e:
        print(f"\n  ❌ Captum test FAILED: {e}")
        all_passed = False
    
    # Test 4: Model wrapper
    try:
        test_model_wrapper()
    except Exception as e:
        print(f"\n  ❌ Model wrapper test FAILED: {e}")
        all_passed = False
    
    # Test 5: Attribution properties
    try:
        test_attribution_properties()
    except Exception as e:
        print(f"\n  ❌ Attribution properties test FAILED: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("\n  ✅ ALL TESTS PASSED!")
        print("\n  You can now run the full IG analysis:")
        print("  python -m src.pipelines.text_based.analyze_integrated_gradients \\")
        print("      --config_filepath <config> \\")
        print("      --checkpoint_path <checkpoint> \\")
        print("      --output_dir ./ig_results")
    else:
        print("\n  ❌ SOME TESTS FAILED")
        print("\n  Please fix the issues before running the full analysis.")
    
    print("\n" + "="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

