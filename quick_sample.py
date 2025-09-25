#!/usr/bin/env python3
"""
Quick sample script to run MicroVelocityAnalyzer on sample data.

This is a simple script for quickly testing the analyzer with default settings.
For comprehensive analysis with all features, use run_sample_analysis.py instead.
"""

import subprocess
import sys
import os

def main():
    """Run a quick analysis on sample data."""
    print("🚀 Quick MicroVelocityAnalyzer Test")
    print("=" * 40)
    
    # Check if sample data exists
    if not os.path.exists("sampledata/sample_allocated.csv"):
        print("❌ Sample data not found!")
        print("   Expected: sampledata/sample_allocated.csv")
        return 1
    
    # Run basic analysis
    print("Running basic analysis...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "micro_velocity_analyzer.micro_velocity_analyzer",
            "--allocated_file", "sampledata/sample_allocated.csv",
            "--transfers_file", "sampledata/sample_transfers.csv", 
            "--output_file", "quick_output.pickle"
        ], timeout=60)
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully!")
            print("📁 Results saved to: quick_output.pickle")
            print("\nTo run comprehensive analysis with all features:")
            print("   python run_sample_analysis.py")
            return 0
        else:
            print(f"❌ Analysis failed with return code {result.returncode}")
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ Analysis timed out")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
