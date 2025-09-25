#!/usr/bin/env python3
"""
Simple test runner for MicroVelocityAnalyzer tests.
"""

import os
import sys
import subprocess

def run_tests():
    """Run all test files in the tests directory."""
    test_dir = os.path.dirname(__file__)
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    print("MicroVelocityAnalyzer Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    for test_file in sorted(test_files):
        test_path = os.path.join(test_dir, test_file)
        print(f"\nRunning {test_file}...")
        
        try:
            result = subprocess.run([sys.executable, test_path], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ {test_file} PASSED")
                if result.stdout.strip():
                    print(result.stdout)
            else:
                print(f"❌ {test_file} FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} TIMEOUT")
            all_passed = False
        except Exception as e:
            print(f"💥 {test_file} ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("💔 SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
