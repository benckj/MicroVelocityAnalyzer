# MicroVelocityAnalyzer Tests

This directory contains test scripts for validating the MicroVelocityAnalyzer functionality.

## Running Tests

### All Tests
```bash
python tests/run_tests.py
```

### Individual Tests
```bash
python tests/test_normalization.py
```

## Test Coverage

### test_normalization.py
Validates the velocity normalization functionality:

1. **Baseline unchanged**: Ensures `--normalize_velocities=none` preserves original behavior
2. **By-balance sanity**: Tests normalization by account balance 
3. **By-supply sanity**: Tests normalization by total supply
4. **Parallel vs single-core parity**: Ensures parallel and single-core execution produce identical results
5. **Error conditions**: Tests proper error handling for invalid inputs

## Test Data
Tests use synthetic data to ensure predictable results:
- Small dataset with known allocations and transfers
- Controlled scenarios for testing specific normalization behaviors
- Deterministic results for comparison between execution modes
