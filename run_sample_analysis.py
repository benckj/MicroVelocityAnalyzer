#!/usr/bin/env python3
"""
Sample analysis script for MicroVelocityAnalyzer.

This script demonstrates how to run the MicroVelocityAnalyzer on sample data
with different configurations, including normalization options and parallel processing.
"""

import os
import sys
import pickle
import subprocess
import numpy as np
from typing import Dict, Tuple, Any

def run_analyzer_command(args: list, description: str) -> Tuple[int, str, str]:
    """Run analyzer with given arguments and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: python -m micro_velocity_analyzer.micro_velocity_analyzer {' '.join(args)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "micro_velocity_analyzer.micro_velocity_analyzer"] + args,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print("❌ Analysis timed out after 120 seconds")
        return 1, "", "Timeout"
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return 1, "", str(e)

def load_and_analyze_results(pickle_file: str) -> Dict[str, Any]:
    """Load pickle results and perform basic analysis."""
    if not os.path.exists(pickle_file):
        print(f"❌ Output file {pickle_file} not found")
        return {}
    
    try:
        with open(pickle_file, 'rb') as f:
            backup_accounts, velocities, balances = pickle.load(f)
        
        stats = {
            'num_addresses': len(velocities),
            'velocity_array_length': len(list(velocities.values())[0]) if velocities else 0,
            'total_velocity': sum(np.sum(v) for v in velocities.values()),
            'max_velocity': max(np.max(v) for v in velocities.values()) if velocities else 0,
            'addresses_with_velocity': sum(1 for v in velocities.values() if np.any(v > 0)),
            'total_final_balance': sum(b[-1] for b in balances.values()) if balances else 0
        }
        
        print(f"\n📊 Analysis Results for {pickle_file}:")
        print(f"   • Number of addresses: {stats['num_addresses']}")
        print(f"   • Velocity array length: {stats['velocity_array_length']}")
        print(f"   • Addresses with non-zero velocity: {stats['addresses_with_velocity']}")
        print(f"   • Total velocity across all addresses: {stats['total_velocity']:.6f}")
        print(f"   • Maximum velocity value: {stats['max_velocity']:.6f}")
        print(f"   • Total final balance: {stats['total_final_balance']:.2f}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error loading results from {pickle_file}: {e}")
        return {}

def compare_results(baseline_stats: Dict[str, Any], normalized_stats: Dict[str, Any], 
                   normalization_type: str) -> None:
    """Compare normalized results with baseline."""
    if not baseline_stats or not normalized_stats:
        return
    
    print(f"\n🔍 Comparison: Baseline vs {normalization_type}")
    print(f"   • Velocity ratio: {normalized_stats['total_velocity'] / baseline_stats['total_velocity']:.6f}")
    print(f"   • Max velocity ratio: {normalized_stats['max_velocity'] / baseline_stats['max_velocity']:.6f}")
    
    if normalization_type == "by_supply":
        expected_ratio = 1.0 / 10000.0  # Expected reduction factor
        actual_ratio = normalized_stats['total_velocity'] / baseline_stats['total_velocity']
        print(f"   • Expected ratio (1/supply): {expected_ratio:.6f}")
        print(f"   • Actual ratio: {actual_ratio:.6f}")
        print(f"   • Ratio difference: {abs(actual_ratio - expected_ratio):.6f}")

def main():
    """Run sample analyses with different configurations."""
    print("🚀 MicroVelocityAnalyzer Sample Analysis Runner")
    print("=" * 60)
    
    # Check if sample data exists
    sample_allocated = "sampledata/sample_allocated.csv"
    sample_transfers = "sampledata/sample_transfers.csv"
    
    if not os.path.exists(sample_allocated) or not os.path.exists(sample_transfers):
        print("❌ Sample data files not found!")
        print(f"   Expected: {sample_allocated}")
        print(f"   Expected: {sample_transfers}")
        print("   Please ensure sample data is available.")
        return 1
    
    # Create output directory
    output_dir = "sample_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for different analysis runs
    analysis_configs = [
        {
            "name": "Baseline Analysis (default)",
            "args": [
                "--allocated_file", sample_allocated,
                "--transfers_file", sample_transfers,
                "--output_file", f"{output_dir}/baseline.pickle"
            ],
            "description": "Standard analysis with no normalization"
        },
        {
            "name": "Balance Normalization",
            "args": [
                "--allocated_file", sample_allocated,
                "--transfers_file", sample_transfers,
                "--output_file", f"{output_dir}/by_balance.pickle",
                "--normalize_velocities", "by_balance"
            ],
            "description": "Velocities normalized by account balances"
        },
        {
            "name": "Supply Normalization",
            "args": [
                "--allocated_file", sample_allocated,
                "--transfers_file", sample_transfers,
                "--output_file", f"{output_dir}/by_supply.pickle",
                "--normalize_velocities", "by_supply",
                "--total_supply", "10000"
            ],
            "description": "Velocities normalized by total supply (10,000)"
        },
        {
            "name": "Parallel Processing (2 cores)",
            "args": [
                "--allocated_file", sample_allocated,
                "--transfers_file", sample_transfers,
                "--output_file", f"{output_dir}/parallel.pickle",
                "--n_cores", "2",
                "--n_chunks", "4"
            ],
            "description": "Parallel processing with 2 cores and 4 chunks"
        },
        {
            "name": "Reduced Temporal Resolution",
            "args": [
                "--allocated_file", sample_allocated,
                "--transfers_file", sample_transfers,
                "--output_file", f"{output_dir}/reduced_resolution.pickle",
                "--save_every_n", "5"
            ],
            "description": "Save every 5th block for reduced memory usage"
        }
    ]
    
    # Store results for comparison
    results_stats = {}
    
    # Run each analysis configuration
    for config in analysis_configs:
        return_code, stdout, stderr = run_analyzer_command(
            config["args"], 
            f"{config['name']}: {config['description']}"
        )
        
        if return_code == 0:
            print("✅ Analysis completed successfully")
            
            # Load and analyze results
            output_file = None
            for i, arg in enumerate(config["args"]):
                if arg == "--output_file" and i + 1 < len(config["args"]):
                    output_file = config["args"][i + 1]
                    break
            
            if output_file:
                stats = load_and_analyze_results(output_file)
                results_stats[config["name"]] = stats
        else:
            print(f"❌ Analysis failed with return code {return_code}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("📊 RESULTS COMPARISON")
    print(f"{'='*60}")
    
    baseline_stats = results_stats.get("Baseline Analysis (default)", {})
    
    if baseline_stats:
        # Compare normalization methods
        balance_stats = results_stats.get("Balance Normalization", {})
        if balance_stats:
            compare_results(baseline_stats, balance_stats, "by_balance")
        
        supply_stats = results_stats.get("Supply Normalization", {})
        if supply_stats:
            compare_results(baseline_stats, supply_stats, "by_supply")
        
        # Check parallel processing consistency
        parallel_stats = results_stats.get("Parallel Processing (2 cores)", {})
        if parallel_stats:
            print(f"\n🔍 Parallel vs Single-core:")
            velocity_diff = abs(parallel_stats['total_velocity'] - baseline_stats['total_velocity'])
            print(f"   • Total velocity difference: {velocity_diff:.10f}")
            if velocity_diff < 1e-9:
                print("   ✅ Parallel processing matches single-core results")
            else:
                print("   ⚠️ Parallel processing differs from single-core")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    successful_runs = len(results_stats)
    total_runs = len(analysis_configs)
    
    print(f"• Total analyses run: {total_runs}")
    print(f"• Successful analyses: {successful_runs}")
    print(f"• Output files saved in: {output_dir}/")
    
    if successful_runs == total_runs:
        print("🎉 All analyses completed successfully!")
        print(f"\nNext steps:")
        print(f"• Examine output files in {output_dir}/")
        print(f"• Compare velocity patterns across different normalizations")
        print(f"• Analyze how normalization affects interpretation of results")
    else:
        print(f"⚠️ {total_runs - successful_runs} analyses failed")
    
    return 0 if successful_runs == total_runs else 1

if __name__ == "__main__":
    sys.exit(main())
