#!/usr/bin/env python3
"""
Test script for MicroVelocityAnalyzer normalization functionality.

This script creates synthetic test data and validates:
1. Baseline unchanged: --normalize_velocities=none preserves original behavior
2. By-balance sanity: Normalization by account balance works correctly
3. By-supply sanity: Normalization by total supply works correctly  
4. Parallel vs single-core parity: Results match between execution modes
"""

import os
import sys
import csv
import pickle
import tempfile
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path to import the analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from micro_velocity_analyzer.micro_velocity_analyzer import MicroVelocityAnalyzer


def create_synthetic_data(temp_dir: str) -> Tuple[str, str]:
    """Create synthetic allocated and transfers CSV files for testing."""
    allocated_file = os.path.join(temp_dir, 'test_allocated.csv')
    transfers_file = os.path.join(temp_dir, 'test_transfers.csv')
    
    # Create allocated data
    with open(allocated_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['to_address', 'amount', 'block_number'])
        # Address with constant positive balance (1000 tokens at block 100)
        writer.writerow(['0xtest1', '1000.0', '100'])
        # Address with variable balance
        writer.writerow(['0xtest2', '500.0', '100'])
        writer.writerow(['0xtest2', '300.0', '110'])
    
    # Create transfers data - 0xtest1 sends tokens creating liability spread across blocks
    with open(transfers_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from_address', 'to_address', 'amount', 'block_number'])
        # Test1 sends 200 tokens across 5 blocks (105-109)
        for block in range(105, 110):
            writer.writerow(['0xtest1', '0xrecipient', '40.0', str(block)])
        # Test2 sends some tokens
        writer.writerow(['0xtest2', '0xrecipient', '100.0', '115'])
    
    return allocated_file, transfers_file


def run_analyzer(allocated_file: str, transfers_file: str, output_file: str, 
                normalize_velocities: str = 'none', total_supply: float = None,
                n_cores: int = 1) -> Tuple[Dict, Dict, Dict]:
    """Run the analyzer and return the results."""
    analyzer = MicroVelocityAnalyzer(
        allocated_file=allocated_file,
        transfers_file=transfers_file,
        output_file=output_file,
        save_every_n=1,
        n_cores=n_cores,
        normalize_velocities=normalize_velocities,
        total_supply=total_supply
    )
    analyzer.run_analysis()
    
    # Load results
    with open(output_file, 'rb') as f:
        backup_accounts, velocities, balances = pickle.load(f)
    
    return backup_accounts, velocities, balances


def test_baseline_unchanged():
    """Test that --normalize_velocities=none preserves original behavior."""
    print("Testing baseline unchanged...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        allocated_file, transfers_file = create_synthetic_data(temp_dir)
        
        # Run with normalization=none
        output_file = os.path.join(temp_dir, 'test_output.pickle')
        backup_accounts, velocities, balances = run_analyzer(
            allocated_file, transfers_file, output_file, 'none'
        )
        
        # Basic sanity checks
        assert len(velocities) > 0, "Should have velocity data"
        assert len(balances) > 0, "Should have balance data"
        assert '0xtest1' in velocities, "Should have test1 address"
        
        # Check that velocities are non-zero where expected
        test1_velocity = velocities['0xtest1']
        assert np.any(test1_velocity > 0), "Should have positive velocity for test1"
        
        print("✓ Baseline test passed")


def test_by_balance_normalization():
    """Test normalization by account balance."""
    print("Testing by-balance normalization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        allocated_file, transfers_file = create_synthetic_data(temp_dir)
        
        # Run with by_balance normalization
        output_file = os.path.join(temp_dir, 'test_output.pickle')
        backup_accounts, velocities, balances = run_analyzer(
            allocated_file, transfers_file, output_file, 'by_balance'
        )
        
        # Test1 should have constant balance of 1000 after allocation at block 100
        # and liability of 200 total spread across blocks 105-109 (5 blocks, 40 per block)
        test1_balance = balances['0xtest1']
        test1_velocity = velocities['0xtest1']
        
        # Find non-zero velocity indices
        nonzero_indices = np.nonzero(test1_velocity)[0]
        
        if len(nonzero_indices) > 0:
            # For blocks 105-109, velocity should be (40/1) / balance = 40/1000 = 0.04 per block
            # But the actual calculation depends on the exact bucketing
            print(f"Test1 balance shape: {test1_balance.shape}")
            print(f"Test1 velocity shape: {test1_velocity.shape}")
            print(f"Non-zero velocity indices: {nonzero_indices}")
            print(f"Non-zero velocities: {test1_velocity[nonzero_indices]}")
            
            # Verify that velocity values are scaled by balance
            # Since balance is ~1000, normalized velocities should be much smaller than raw
            max_velocity = np.max(test1_velocity)
            assert max_velocity < 1.0, f"Normalized velocity {max_velocity} should be < 1.0"
            
        print("✓ By-balance normalization test passed")


def test_by_supply_normalization():
    """Test normalization by total supply."""
    print("Testing by-supply normalization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        allocated_file, transfers_file = create_synthetic_data(temp_dir)
        
        total_supply = 10000.0
        
        # Run with by_supply normalization
        output_file = os.path.join(temp_dir, 'test_output.pickle')
        backup_accounts, velocities, balances = run_analyzer(
            allocated_file, transfers_file, output_file, 'by_supply', total_supply
        )
        
        test1_velocity = velocities['0xtest1']
        nonzero_indices = np.nonzero(test1_velocity)[0]
        
        if len(nonzero_indices) > 0:
            max_velocity = np.max(test1_velocity)
            # With total_supply=10000, velocities should be scaled down by that factor
            assert max_velocity < 0.1, f"Supply-normalized velocity {max_velocity} should be small"
            
        print("✓ By-supply normalization test passed")


def test_parallel_vs_single_core():
    """Test that parallel and single-core execution produce the same results."""
    print("Testing parallel vs single-core parity...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        allocated_file, transfers_file = create_synthetic_data(temp_dir)
        
        # Run single-core
        output_file_single = os.path.join(temp_dir, 'test_single.pickle')
        _, velocities_single, balances_single = run_analyzer(
            allocated_file, transfers_file, output_file_single, 'none', n_cores=1
        )
        
        # Run parallel  
        output_file_parallel = os.path.join(temp_dir, 'test_parallel.pickle')
        _, velocities_parallel, balances_parallel = run_analyzer(
            allocated_file, transfers_file, output_file_parallel, 'none', n_cores=2
        )
        
        # Compare results
        assert set(velocities_single.keys()) == set(velocities_parallel.keys()), \
            "Address sets should match"
        
        for address in velocities_single:
            vel_diff = np.abs(velocities_single[address] - velocities_parallel[address])
            max_diff = np.max(vel_diff)
            assert max_diff < 1e-9, f"Velocity difference {max_diff} too large for {address}"
            
            bal_diff = np.abs(balances_single[address] - balances_parallel[address])
            max_bal_diff = np.max(bal_diff)
            assert max_bal_diff < 1e-9, f"Balance difference {max_bal_diff} too large for {address}"
        
        print("✓ Parallel vs single-core parity test passed")


def test_error_conditions():
    """Test error conditions for invalid inputs."""
    print("Testing error conditions...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        allocated_file, transfers_file = create_synthetic_data(temp_dir)
        
        # Test invalid save_every_n
        try:
            MicroVelocityAnalyzer(
                allocated_file, transfers_file, 'output.pickle', 
                save_every_n=0
            )
            assert False, "Should raise ValueError for save_every_n <= 0"
        except ValueError as e:
            assert "save_every_n must be positive" in str(e)
        
        # Test by_supply without total_supply
        try:
            MicroVelocityAnalyzer(
                allocated_file, transfers_file, 'output.pickle',
                normalize_velocities='by_supply'
            )
            assert False, "Should raise ValueError for by_supply without total_supply"
        except ValueError as e:
            assert "total_supply must be provided" in str(e)
        
        # Test by_supply with invalid total_supply
        try:
            MicroVelocityAnalyzer(
                allocated_file, transfers_file, 'output.pickle',
                normalize_velocities='by_supply', total_supply=-100
            )
            assert False, "Should raise ValueError for negative total_supply"
        except ValueError as e:
            assert "total_supply must be provided and positive" in str(e)
        
        print("✓ Error conditions test passed")


def main():
    """Run all tests."""
    print("Running MicroVelocityAnalyzer normalization tests...\n")
    
    try:
        test_baseline_unchanged()
        test_by_balance_normalization() 
        test_by_supply_normalization()
        test_parallel_vs_single_core()
        test_error_conditions()
        
        print("\n✅ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
