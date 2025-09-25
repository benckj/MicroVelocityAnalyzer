# Sample Analysis Guide

This document explains how to run the MicroVelocityAnalyzer on sample data using the provided `run_sample_analysis.py` script.

## Quick Start

```bash
# Run all sample analyses
python run_sample_analysis.py

# Or make it executable and run directly
chmod +x run_sample_analysis.py
./run_sample_analysis.py
```

## What the Script Does

The `run_sample_analysis.py` script demonstrates five different analysis configurations:

### 1. **Baseline Analysis** (Default)
- **Command**: Standard analysis with no normalization
- **Purpose**: Establishes baseline velocity measurements
- **Output**: `sample_outputs/baseline.pickle`

### 2. **Balance Normalization**
- **Command**: `--normalize_velocities by_balance`
- **Purpose**: Normalizes velocities by account balance at each time point
- **Output**: `sample_outputs/by_balance.pickle`
- **Use Case**: Compare velocity relative to account wealth

### 3. **Supply Normalization**
- **Command**: `--normalize_velocities by_supply --total_supply 10000`
- **Purpose**: Normalizes velocities by total token supply (constant)
- **Output**: `sample_outputs/by_supply.pickle`
- **Use Case**: Express velocity as fraction of total supply turnover

### 4. **Parallel Processing**
- **Command**: `--n_cores 2 --n_chunks 4`
- **Purpose**: Demonstrates parallel processing capabilities
- **Output**: `sample_outputs/parallel.pickle`
- **Verification**: Should match single-core results exactly

### 5. **Reduced Temporal Resolution**
- **Command**: `--save_every_n 5`
- **Purpose**: Saves every 5th block to reduce memory usage
- **Output**: `sample_outputs/reduced_resolution.pickle`
- **Trade-off**: Lower memory usage vs. temporal precision

## Understanding the Results

### Key Metrics Reported

- **Number of addresses**: Total addresses processed
- **Velocity array length**: Time points in analysis (depends on `save_every_n`)
- **Addresses with non-zero velocity**: Addresses that had token transfers
- **Total velocity**: Sum of all velocity values across time and addresses
- **Maximum velocity**: Highest velocity value observed
- **Total final balance**: Sum of all account balances at the end

### Normalization Effects

#### Supply Normalization Results
```
• Expected ratio (1/supply): 0.000100
• Actual ratio: 0.000100
• Ratio difference: 0.000000
```
✅ **Perfect**: Supply normalization reduces velocities by exactly 1/total_supply

#### Parallel Processing Verification
```
• Total velocity difference: 0.0000000000
✅ Parallel processing matches single-core results
```
✅ **Perfect**: Parallel and single-core execution produce identical results

## Manual Analysis Examples

You can also run individual analyses manually:

```bash
# Basic analysis
python -m micro_velocity_analyzer.micro_velocity_analyzer \
    --allocated_file sampledata/sample_allocated.csv \
    --transfers_file sampledata/sample_transfers.csv \
    --output_file my_output.pickle

# With balance normalization
python -m micro_velocity_analyzer.micro_velocity_analyzer \
    --allocated_file sampledata/sample_allocated.csv \
    --transfers_file sampledata/sample_transfers.csv \
    --output_file normalized_output.pickle \
    --normalize_velocities by_balance

# With supply normalization
python -m micro_velocity_analyzer.micro_velocity_analyzer \
    --allocated_file sampledata/sample_allocated.csv \
    --transfers_file sampledata/sample_transfers.csv \
    --output_file supply_normalized.pickle \
    --normalize_velocities by_supply \
    --total_supply 1000000

# Parallel processing with 4 cores
python -m micro_velocity_analyzer.micro_velocity_analyzer \
    --allocated_file sampledata/sample_allocated.csv \
    --transfers_file sampledata/sample_transfers.csv \
    --output_file parallel_output.pickle \
    --n_cores 4 \
    --n_chunks 8
```

## Analyzing Results

### Loading Results in Python

```python
import pickle
import numpy as np

# Load results
with open('sample_outputs/baseline.pickle', 'rb') as f:
    backup_accounts, velocities, balances = pickle.load(f)

# Examine velocity patterns
for address, velocity in velocities.items():
    if np.any(velocity > 0):
        print(f"Address {address}: max velocity = {np.max(velocity):.6f}")
        
# Compare balances vs velocities
for address in velocities.keys():
    final_balance = balances[address][-1]
    total_velocity = np.sum(velocities[address])
    print(f"{address}: Balance={final_balance:.2f}, Velocity={total_velocity:.6f}")
```

### Comparative Analysis

```python
# Compare normalization effects
baseline_vel = np.sum([np.sum(v) for v in baseline_velocities.values()])
supply_vel = np.sum([np.sum(v) for v in supply_velocities.values()])
reduction_factor = supply_vel / baseline_vel
print(f"Supply normalization reduces velocity by factor: {reduction_factor:.6f}")
```

## Output Files

All output files are saved in the `sample_outputs/` directory:

- `baseline.pickle` - Standard analysis results
- `by_balance.pickle` - Balance-normalized results  
- `by_supply.pickle` - Supply-normalized results
- `parallel.pickle` - Parallel processing results
- `reduced_resolution.pickle` - Lower temporal resolution results

Each pickle file contains: `[backup_accounts, velocities, balances]`

## Next Steps

1. **Examine the output files** to understand velocity patterns
2. **Compare normalization effects** on interpretation 
3. **Adapt the script** for your own data and requirements
4. **Scale up** using parallel processing for larger datasets
