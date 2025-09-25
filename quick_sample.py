#!/usr/bin/env python3
"""
Quick sample script to run MicroVelocityAnalyzer on sample data.

This is a simple script for quickly testing the analyzer with default settings.
For comprehensive analysis with all features, use run_sample_analysis.py instead.
"""

import subprocess
import sys
import os

def run_analysis(name, output_file, extra_args=None):
    """Run a single analysis with given parameters."""
    print(f"\n📊 Running {name}...")
    
    cmd = [
        sys.executable, "-m", "micro_velocity_analyzer.micro_velocity_analyzer",
        "--allocated_file", "sampledata/sample_allocated.csv",
        "--transfers_file", "sampledata/sample_transfers.csv", 
        "--output_file", output_file
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        result = subprocess.run(cmd, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ {name} completed successfully!")
            print(f"📁 Results saved to: {output_file}")
            return True
        else:
            print(f"❌ {name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {name} timed out")
        return False
    except Exception as e:
        print(f"❌ {name} error: {e}")
        return False

def load_and_compare_results():
    """Load and compare the two analysis results."""
    import pickle
    import numpy as np
    
    try:
        # Load baseline results
        with open('quick_baseline.pickle', 'rb') as f:
            _, baseline_velocities, baseline_balances = pickle.load(f)
        
        # Load normalized results  
        with open('quick_normalized.pickle', 'rb') as f:
            _, normalized_velocities, normalized_balances = pickle.load(f)
        
        print("\n🔍 COMPARISON RESULTS:")
        print("=" * 40)
        
        # Calculate totals
        baseline_total = sum(np.sum(v) for v in baseline_velocities.values())
        normalized_total = sum(np.sum(v) for v in normalized_velocities.values())
        baseline_max = max(np.max(v) for v in baseline_velocities.values())
        normalized_max = max(np.max(v) for v in normalized_velocities.values())
        
        print(f"📈 Baseline Analysis:")
        print(f"   • Total velocity: {baseline_total:.6f}")
        print(f"   • Max velocity: {baseline_max:.6f}")
        print(f"   • Addresses processed: {len(baseline_velocities)}")
        
        print(f"\n📉 Balance-Normalized Analysis:")
        print(f"   • Total velocity: {normalized_total:.6f}")
        print(f"   • Max velocity: {normalized_max:.6f}")
        print(f"   • Addresses processed: {len(normalized_velocities)}")
        
        if baseline_total > 0:
            ratio = normalized_total / baseline_total
            print(f"\n🔢 Normalization Effect:")
            print(f"   • Velocity reduction ratio: {ratio:.6f}")
            print(f"   • Balance normalization scales velocities by account wealth")
        
        return True
        
    except Exception as e:
        print(f"❌ Error comparing results: {e}")
        return False

def main():
    """Run quick analysis with both baseline and balance normalization."""
    print("🚀 Quick MicroVelocityAnalyzer Comparison")
    print("=" * 50)
    
    # Check if sample data exists
    if not os.path.exists("sampledata/sample_allocated.csv"):
        print("❌ Sample data not found!")
        print("   Expected: sampledata/sample_allocated.csv")
        return 1
    
    success_count = 0
    
    # Run baseline analysis (no normalization)
    if run_analysis("Baseline Analysis", "quick_baseline.pickle"):
        success_count += 1
    
    # Run balance-normalized analysis
    if run_analysis("Balance-Normalized Analysis", "quick_normalized.pickle", 
                   ["--normalize_velocities", "by_balance"]):
        success_count += 1
    
    # Compare results if both succeeded
    if success_count == 2:
        print("\n" + "=" * 50)
        load_and_compare_results()
        print("\n🎉 Both analyses completed successfully!")
        print("📁 Output files:")
        print("   • quick_baseline.pickle (no normalization)")
        print("   • quick_normalized.pickle (balance normalization)")
    else:
        print(f"\n⚠️ Only {success_count}/2 analyses succeeded")
    
    print("\nFor comprehensive analysis with all features:")
    print("   python run_sample_analysis.py")
    
    return 0 if success_count == 2 else 1

if __name__ == "__main__":
    sys.exit(main())
