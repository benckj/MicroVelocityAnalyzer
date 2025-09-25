#!/usr/bin/env python3
"""
MicroVelocityAnalyzer: A tool for analyzing token velocity in blockchain transactions.

This module provides functionality to analyze micro-velocity patterns from token 
allocation and transfer data. It supports optional velocity normalization and 
parallel processing for large datasets.

Author: Francesco Maria De Collibus
Email: francesco.decollibus@business.uzh.ch
"""

import argparse
import os
import pickle
import copy
import numpy as np
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Union

def process_chunk_balances(args):
    """Process balance calculations for a chunk of addresses (legacy version).
    
    This function calculates running balances for each address by applying
    balance changes (assets positive, liabilities negative) at each save point.
    
    Args:
        args: Tuple containing (addresses, accounts_chunk, min_block_number, 
              max_block_number, save_every_n, LIMIT, pos)
              
    Returns:
        Dict[str, np.ndarray]: Mapping of addresses to their balance arrays
    """
    addresses, accounts_chunk, min_block_number, max_block_number, save_every_n, LIMIT, pos = args
    results = {}
    
    # Create save grid: blocks where we save balance snapshots
    save_block_numbers = [min_block_number + i * save_every_n for i in range(LIMIT)]
    
    for address in tqdm(addresses, position=pos, leave=False):
        # Collect all balance changes for this address
        balance_changes = []
        
        # Add asset transactions (positive balance changes)
        for block_number, amount in accounts_chunk[address][0].items():
            balance_changes.append((int(block_number), float(amount)))
            
        # Add liability transactions (negative balance changes)  
        for block_number, amount in accounts_chunk[address][1].items():
            balance_changes.append((int(block_number), -float(amount)))
            
        # Sort balance changes chronologically
        balance_changes.sort()
        
        # Calculate running balance at each save point
        balance = 0.0
        change_idx = 0
        balances = []
        
        # Sweep through save points, applying all changes up to each point
        for block_number in save_block_numbers:
            # Apply all balance changes that occurred by this block
            while change_idx < len(balance_changes) and balance_changes[change_idx][0] <= block_number:
                balance += balance_changes[change_idx][1]
                change_idx += 1
            balances.append(balance)
            
        results[address] = np.array(balances, dtype=np.float64)
    
    # Clean up memory
    del balance_changes, balances
    return results

def process_chunk_balances_v2(args):
    """Process balance chunks with corrected sweep logic (improved version).
    
    This is the corrected version that properly sweeps the save grid without
    skipping index 0, ensuring accurate balance computation at all save points.
    
    Args:
        args: Tuple containing (addresses, accounts_chunk, min_block_number, 
              max_block_number, save_every_n, LIMIT, pos)
              
    Returns:
        Dict[str, np.ndarray]: Mapping of addresses to their balance arrays
    """
    addresses, accounts_chunk, min_block_number, max_block_number, save_every_n, LIMIT, pos = args
    
    # Create save grid using numpy for efficiency
    save_block_numbers = np.arange(LIMIT, dtype=np.int64) * save_every_n + min_block_number
    results = {}

    for address in tqdm(addresses, position=pos, leave=False):
        # Collect all balance changes for this address
        balance_changes = []
        
        # Add asset transactions (positive balance changes)
        for block_number, amount in accounts_chunk[address][0].items():
            balance_changes.append((int(block_number), float(amount)))
            
        # Add liability transactions (negative balance changes)
        for block_number, amount in accounts_chunk[address][1].items():
            balance_changes.append((int(block_number), -float(amount)))
            
        # Sort balance changes chronologically
        balance_changes.sort()
        
        # Sweep through save grid, carrying balance forward correctly
        balance = 0.0
        change_idx = 0
        balances = np.zeros(LIMIT, dtype=np.float64)
        
        # For each save point, apply all changes up to that block
        for i, bn in enumerate(save_block_numbers):
            # Apply all balance changes that occurred by this save point
            while change_idx < len(balance_changes) and balance_changes[change_idx][0] <= bn:
                balance += balance_changes[change_idx][1]
                change_idx += 1
            balances[i] = balance

        results[address] = balances
    
    return results

def process_chunk_velocities(args):
    """Process velocity calculations for a chunk of addresses with optional normalization.
    
    This function implements the core velocity matching algorithm, where assets and
    liabilities are matched chronologically to calculate token velocity. Supports
    optional normalization by balance or total supply.
    
    Velocity Matching Algorithm:
    1. For each liability (outgoing transfer), find the most recent asset (incoming)
    2. Match them partially or fully based on available amounts
    3. Calculate velocity as amount/duration for the time span
    4. Apply normalization if enabled
    
    Args:
        args: Tuple containing (addresses, accounts_chunk, min_block_number, 
              save_every_n, LIMIT, pos, normalize_velocities, total_supply, balances_chunk)
              
    Returns:
        Dict[str, np.ndarray]: Mapping of addresses to their velocity arrays
    """
    addresses, accounts_chunk, min_block_number, save_every_n, LIMIT, pos, normalize_velocities, total_supply, balances_chunk = args
    results = {}
    
    for address in tqdm(addresses, position=pos, leave=False):
        # Only process addresses that have both assets and liabilities
        if len(accounts_chunk[address][0]) > 0 and len(accounts_chunk[address][1]) > 0:
            # Separate and sort asset and liability transaction keys
            arranged_keys = [list(accounts_chunk[address][0].keys()), list(accounts_chunk[address][1].keys())]
            arranged_keys[0].sort()  # Assets (incoming transfers)
            arranged_keys[1].sort()  # Liabilities (outgoing transfers)
            
            # Initialize velocity array for this address
            ind_velocity = np.zeros(LIMIT, dtype=np.float64)

            # Velocity matching algorithm: process each liability (outgoing transfer)
            for border in arranged_keys[1]:  # border = liability block number
                # Refresh asset keys (they get modified during matching)
                arranged_keys[0] = list(accounts_chunk[address][0].keys())
                test = np.array(arranged_keys[0], dtype=int)

                # Find assets that occurred before this liability
                # Process from most recent to oldest (LIFO matching)
                for i in range(0, len(test[test < border])):
                    # Get the i-th most recent asset before this liability
                    counter = test[test < border][(len(test[test < border]) - 1) - i]
                    
                    # Get amounts for matching
                    asset_amount = float(accounts_chunk[address][0][counter])
                    liability_amount = float(accounts_chunk[address][1][border])
                    
                    # Determine matching scenario
                    if (asset_amount - liability_amount) >= 0:
                        # Asset amount >= liability amount: asset can fully cover liability
                        idx_range = np.unique(np.arange(counter - min_block_number, border - min_block_number)//save_every_n)
                        
                        if len(idx_range) == 1:
                            # Same bucket: no velocity calculation needed
                            accounts_chunk[address][0][counter] -= liability_amount
                            accounts_chunk[address][1].pop(border)
                            break
                        else:
                            # Different buckets: calculate velocity over time span
                            duration = border - counter
                            if duration > 0:
                                # Base velocity calculation: amount/time
                                added = liability_amount / duration
                                
                                # Apply normalization if enabled
                                if normalize_velocities == 'by_balance':
                                    # Normalize by account balance at each bucket
                                    denom = np.maximum(1e-12, balances_chunk[address][idx_range[1:]])
                                    added = added / denom
                                elif normalize_velocities == 'by_supply':
                                    # Normalize by total supply (constant)
                                    denom = max(1e-12, total_supply)
                                    added = added / denom
                                
                                # Add velocity to affected buckets (skip first bucket)
                                ind_velocity[idx_range[1:]] += added
                                
                            # Update remaining amounts after matching
                            accounts_chunk[address][0][counter] -= liability_amount
                            accounts_chunk[address][1].pop(border)
                            break
                    else:
                        # Asset amount < liability amount: asset partially covers liability
                        idx_range = np.unique(np.arange(counter - min_block_number, border - min_block_number)//save_every_n)
                        
                        if len(idx_range) == 1:
                            # Same bucket: no velocity calculation needed
                            accounts_chunk[address][1][border] -= asset_amount
                            accounts_chunk[address][0].pop(counter)
                        else:
                            # Different buckets: calculate velocity for partial match
                            duration = border - counter
                            if duration > 0:
                                # Base velocity calculation: asset_amount/time
                                added = asset_amount / duration
                                
                                # Apply normalization if enabled
                                if normalize_velocities == 'by_balance':
                                    # Normalize by account balance at each bucket
                                    denom = np.maximum(1e-12, balances_chunk[address][idx_range[1:]])
                                    added = added / denom
                                elif normalize_velocities == 'by_supply':
                                    # Normalize by total supply (constant)
                                    denom = max(1e-12, total_supply)
                                    added = added / denom
                                
                                # Add velocity to affected buckets (skip first bucket)
                                ind_velocity[idx_range[1:]] += added
                                
                            # Update remaining amounts after partial matching
                            accounts_chunk[address][1][border] -= asset_amount
                            accounts_chunk[address][0].pop(counter)
            
            # Store computed velocity array for this address            
            results[address] = ind_velocity
    return results

class MicroVelocityAnalyzer:
    def __init__(self, allocated_file: str, transfers_file: str, output_file: str = 'temp/general_velocities.pickle', 
                 save_every_n: int = 1, n_cores: int = 1, n_chunks: int = 1, 
                 split_save: bool = False, batch_size: int = 1, 
                 normalize_velocities: str = 'none', total_supply: Optional[float] = None):
        """Initialize MicroVelocityAnalyzer with optional velocity normalization.
        
        Args:
            allocated_file: Path to allocated CSV file
            transfers_file: Path to transfers CSV file
            output_file: Path to output pickle file
            save_every_n: Save every Nth position of the velocity array
            n_cores: Number of cores to use
            n_chunks: Number of chunks to split data into
            split_save: Split save into different files
            batch_size: Number of chunks to process in a single batch
            normalize_velocities: Normalization mode ('none', 'by_balance', 'by_supply')
            total_supply: Total supply for 'by_supply' normalization
        """
        if save_every_n <= 0:
            raise ValueError("save_every_n must be positive")
        
        if normalize_velocities == 'by_supply':
            if total_supply is None or total_supply <= 0:
                raise ValueError("total_supply must be provided and positive when normalize_velocities='by_supply'")
        
        self.allocated_file = allocated_file
        self.transfers_file = transfers_file
        self.output_file = output_file
        self.save_every_n = save_every_n
        self.n_cores = n_cores
        self.n_chunks = n_chunks
        self.split_save = split_save
        self.batch_size = batch_size
        self.normalize_velocities = normalize_velocities
        self.total_supply = total_supply if total_supply is not None else 0.0
        
        self.accounts: Dict[str, List[Dict[int, float]]] = {}
        self.backup_accounts: Dict[str, List[Dict[int, float]]] = {}
        self.min_block_number = float('inf')
        self.max_block_number = float('-inf')
        self.velocities: Dict[str, np.ndarray] = {}
        self.balances: Dict[str, np.ndarray] = {}
        self.LIMIT = 0
        self._create_output_folder()

    def _create_output_folder(self) -> None:
        """Create output directory if it doesn't exist."""
        output_folder = os.path.dirname(self.output_file)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def load_allocated_data(self) -> None:
        """Load initial token allocations from CSV file.
        
        Processes allocations (initial token distributions) and adds them as
        asset transactions to the accounts structure.
        """
        with open(self.allocated_file, 'r') as file:
            reader = csv.DictReader(file)
            for line in tqdm(reader):
                self._process_allocation(line)

    def _process_allocation(self, line: Dict[str, str]) -> None:
        """Process a single allocation line from CSV."""
        to_address = line['to_address'].lower()
        try:
            amount = float(line['amount'])  # Use float
            block_number = int(line['block_number'])
        except ValueError:
            print(f"Invalid data in allocated_file: {line}")
            return  # Skip this line

        if to_address not in self.accounts:
            self.accounts[to_address] = [{}, {}]
        
        if block_number not in self.accounts[to_address][0]:
            self.accounts[to_address][0][block_number] = amount
        else:
            self.accounts[to_address][0][block_number] += amount

        self.min_block_number = min(self.min_block_number, block_number)
        self.max_block_number = max(self.max_block_number, block_number)

    def load_transfer_data(self) -> None:
        """Load transfer transactions from CSV file.
        
        Processes transfers between addresses, creating both asset entries
        (for recipients) and liability entries (for senders).
        """
        with open(self.transfers_file, 'r') as file:
            reader = csv.DictReader(file)
            for line in tqdm(reader):
                self._process_transfer(line)

    def _process_transfer(self, line: Dict[str, str]) -> None:
        """Process a single transfer line from CSV."""
        from_address = line['from_address'].lower()
        to_address = line['to_address'].lower()
        try:
            amount = float(line['amount'])  # Use float
            block_number = int(line['block_number'])
        except ValueError:
            print(f"Invalid data in transfers_file: {line}")
            return  # Skip this line

        # Assets
        if to_address not in self.accounts:
            self.accounts[to_address] = [{}, {}]
        if block_number not in self.accounts[to_address][0]:
            self.accounts[to_address][0][block_number] = amount
        else:
            self.accounts[to_address][0][block_number] += amount

        # Liabilities
        if from_address not in self.accounts:
            self.accounts[from_address] = [{}, {}]
        if block_number not in self.accounts[from_address][1]:
            self.accounts[from_address][1][block_number] = amount
        else:
            self.accounts[from_address][1][block_number] += amount

        self.min_block_number = min(self.min_block_number, block_number)
        self.max_block_number = max(self.max_block_number, block_number)

    def calculate_balances(self) -> None:
        """Calculate running balances for all addresses (single-core version).
        
        For each address, computes the balance at each save point by applying
        all balance changes (assets positive, liabilities negative) chronologically.
        """
        # Create save grid: blocks where we save balance snapshots
        save_block_numbers = [self.min_block_number + i * self.save_every_n for i in range(self.LIMIT)]
        
        for address in tqdm(self.accounts.keys()):
            # Collect all balance changes for this address
            balance_changes = []
            
            # Add asset transactions (positive balance changes)
            for block_number, amount in self.accounts[address][0].items():
                balance_changes.append((int(block_number), float(amount)))
                
            # Add liability transactions (negative balance changes)
            for block_number, amount in self.accounts[address][1].items():
                balance_changes.append((int(block_number), -float(amount)))
                
            # Sort balance changes chronologically
            balance_changes.sort()
            
            # Calculate running balance at each save point
            balance = 0.0
            change_idx = 0
            balances = []
            
            # Sweep through save points, applying all changes up to each point
            for block_number in save_block_numbers:
                # Apply all balance changes that occurred by this block
                while change_idx < len(balance_changes) and balance_changes[change_idx][0] <= block_number:
                    balance += balance_changes[change_idx][1]
                    change_idx += 1
                balances.append(balance)
                
            self.balances[address] = np.array(balances, dtype=np.float64)
        
    # def calculate_balances_parallel(self):
    #     addresses = list(self.accounts.keys())
    #     np.random.shuffle(addresses) # Shuffle to avoid having a few addresses with many transactions in the same chunk
    #     chunk_size = max(1, len(addresses) // self.n_chunks)
    #     chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]

    #     # Process in batches of n_cores
    #     total_chunks = len(chunks)
    #     processed_chunks = 0
        
    #     with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
    #         with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
    #             while processed_chunks < total_chunks:
    #                 # Submit batch of n_cores chunks
    #                 current_batch = chunks[processed_chunks:processed_chunks + self.n_cores]
    #                 futures = []
                    
    #                 for i, chunk in enumerate(current_batch):
    #                     accounts_chunk = {address: self.accounts[address] for address in chunk}
    #                     args = (chunk, accounts_chunk, self.min_block_number, 
    #                         self.max_block_number, self.save_every_n, 
    #                         self.LIMIT, i + 1)
    #                     futures.append(executor.submit(process_chunk_balances_v2, args))
                    
    #                 # Process results as they complete
    #                 for future in as_completed(futures):
    #                     chunk_results = future.result()
    #                     self.balances.update(chunk_results)
    #                     del chunk_results
    #                     processed_chunks += 1
    #                     pbar.update(1)
                    
    #                 # Clean up
    #                 del futures

    def _get_split_filename(self, base_type: str, last_address: str) -> str:
        """Generate filename for split saves.
        
        Args:
            base_type: Type of data ('balances' or 'velocities')
            last_address: Last address in the chunk for unique naming
            
        Returns:
            str: Generated filename for split save
        """
        dirname = os.path.dirname(self.output_file)
        basename = os.path.splitext(os.path.basename(self.output_file))[0]
        return os.path.join(dirname, f"{basename}_{base_type}_{last_address}.pickle")

    def _save_split_results(self, results: Dict, result_type: str, last_address: str) -> None:
        """Save intermediate results to split file.
        
        Args:
            results: Results dictionary to save
            result_type: Type of results ('balances' or 'velocities')
            last_address: Last address in chunk for filename generation
        """
        filename = self._get_split_filename(result_type, last_address)
        print(f'Saving {result_type} to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def calculate_balances_parallel(self) -> None:
        """Calculate running balances for all addresses using parallel processing.
        
        Splits addresses into chunks and processes them in parallel using
        multiple worker processes. Supports both in-memory and split-file saving.
        """
        addresses = list(self.accounts.keys())
        
        # Shuffle addresses to distribute workload evenly (unless using split save)
        if not self.split_save:
            np.random.shuffle(addresses)
            
        # Split addresses into chunks for parallel processing
        chunk_size = max(1, len(addresses) // self.n_chunks)
        chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]
        
        total_chunks = len(chunks)
        processed_chunks = 0
        batch_results = {}
        
        # Process chunks in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                while processed_chunks < total_chunks:
                    # Create batch of chunks to process simultaneously
                    current_batch = chunks[processed_chunks:processed_chunks + self.n_cores*self.batch_size]
                    futures = []
                    
                    # Submit each chunk for processing
                    for i, chunk in enumerate(current_batch):
                        # Create subset of accounts for this chunk
                        accounts_chunk = {address: self.accounts[address] for address in chunk}
                        args = (chunk, accounts_chunk, self.min_block_number, 
                               self.max_block_number, self.save_every_n, 
                               self.LIMIT, i + 1)
                        futures.append(executor.submit(process_chunk_balances_v2, args))
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        batch_results.update(chunk_results)
                        processed_chunks += 1
                        pbar.update(1)
                    
                    # Save results based on split_save setting
                    if self.split_save:
                        # Save to separate file for this batch
                        last_address = current_batch[-1][-1]
                        self._save_split_results(batch_results, 'balances', last_address)
                    else:
                        # Accumulate in main balances dictionary
                        self.balances.update(batch_results)
                    
                    # Clear batch results unconditionally (important fix)
                    batch_results = {}
                    
                    # Clean up futures list
                    del futures

    def calculate_velocities_parallel(self) -> None:
        """Calculate velocities for all addresses using parallel processing.
        
        Splits addresses into chunks and processes them in parallel. This is where
        the core velocity matching algorithm runs, with optional normalization support.
        Balances must be computed before this step if using by_balance normalization.
        """
        addresses = list(self.accounts.keys())
        
        # Shuffle addresses to distribute workload evenly (unless using split save)
        if not self.split_save:
            np.random.shuffle(addresses)
            
        # Split addresses into chunks for parallel processing
        chunk_size = max(1, len(addresses) // self.n_chunks)
        chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]
        
        batch_results = {}
        processed_chunks = 0
        
        # Process chunks in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                while processed_chunks < len(chunks):
                    # Create batch of chunks to process simultaneously
                    current_batch = chunks[processed_chunks:processed_chunks + self.n_cores*self.batch_size]
                    futures = []
                    
                    # Submit each chunk for processing
                    for i, chunk in enumerate(current_batch):
                        # Create subset of accounts for this chunk
                        accounts_chunk = {address: self.accounts[address] for address in chunk}
                        
                        # Prepare balances chunk for by_balance normalization
                        balances_chunk = {}
                        if self.normalize_velocities == 'by_balance':
                            # Only include balances for addresses in this chunk
                            balances_chunk = {address: self.balances[address] for address in chunk if address in self.balances}
                        
                        # Prepare arguments including normalization parameters
                        args = (chunk, accounts_chunk, self.min_block_number, 
                               self.save_every_n, self.LIMIT, i + 1, 
                               self.normalize_velocities, self.total_supply, balances_chunk)
                        futures.append(executor.submit(process_chunk_velocities, args))
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        batch_results.update(chunk_results)
                        processed_chunks += 1
                        pbar.update(1)
                    
                    # Save results based on split_save setting
                    if self.split_save:
                        # Save to separate file for this batch
                        last_address = current_batch[-1][-1]
                        self._save_split_results(batch_results, 'velocities', last_address)
                    else:
                        # Accumulate in main velocities dictionary
                        self.velocities.update(batch_results)
                    
                    # Clear batch results unconditionally (important fix)
                    batch_results = {}
                    
                    # Clean up futures list
                    del futures

    def calculate_velocities(self) -> None:
        """Calculate velocities for all addresses (single-core version).
        
        Processes each address individually using the velocity matching algorithm.
        Only addresses with both assets and liabilities are processed.
        """
        for address in tqdm(self.accounts.keys()):
            # Only process addresses that have both incoming and outgoing transactions
            if len(self.accounts[address][0]) > 0 and len(self.accounts[address][1]) > 0:
                self._calculate_individual_velocity(address)

    def _calculate_individual_velocity(self, address: str) -> None:
        """Calculate individual velocity for a single address with optional normalization."""
        arranged_keys = [list(self.accounts[address][0].keys()), list(self.accounts[address][1].keys())]
        arranged_keys[0].sort()
        arranged_keys[1].sort()
        ind_velocity = np.zeros(self.LIMIT, dtype=np.float64)

        for border in tqdm(arranged_keys[1], leave=False):
            arranged_keys[0] = list(self.accounts[address][0].keys())
            test = np.array(arranged_keys[0], dtype=int)

            for i in range(0, len(test[test < border])):
                counter = test[test < border][(len(test[test < border]) - 1) - i]
                asset_amount = float(self.accounts[address][0][counter])
                liability_amount = float(self.accounts[address][1][border])
                if (asset_amount - liability_amount) >= 0:
                    idx_range = np.unique(np.arange(counter - self.min_block_number, border - self.min_block_number)//self.save_every_n)
                    if len(idx_range) == 1:
                        self.accounts[address][0][counter] -= liability_amount
                        self.accounts[address][1].pop(border)
                        break
                    else:
                        duration = border - counter
                        if duration > 0:
                            added = liability_amount / duration
                            # Apply normalization if enabled
                            if self.normalize_velocities == 'by_balance':
                                denom = np.maximum(1e-12, self.balances[address][idx_range[1:]])
                                added = added / denom
                            elif self.normalize_velocities == 'by_supply':
                                denom = max(1e-12, self.total_supply)
                                added = added / denom
                            ind_velocity[idx_range[1:]] += added
                        self.accounts[address][0][counter] -= liability_amount
                        self.accounts[address][1].pop(border)
                        break
                else:
                    idx_range = np.unique(np.arange(counter - self.min_block_number, border - self.min_block_number)//self.save_every_n)
                    if len(idx_range) == 1:
                        self.accounts[address][1][border] -= asset_amount
                        self.accounts[address][0].pop(counter)
                    else:
                        duration = border - counter
                        if duration > 0:
                            added = asset_amount / duration
                            # Apply normalization if enabled
                            if self.normalize_velocities == 'by_balance':
                                denom = np.maximum(1e-12, self.balances[address][idx_range[1:]])
                                added = added / denom
                            elif self.normalize_velocities == 'by_supply':
                                denom = max(1e-12, self.total_supply)
                                added = added / denom
                            ind_velocity[idx_range[1:]] += added
                        self.accounts[address][1][border] -= asset_amount
                        self.accounts[address][0].pop(counter)
        self.velocities[address] = ind_velocity

    # def calculate_velocities_parallel(self):
    #     addresses = list(self.accounts.keys())
    #     np.random.shuffle(addresses) # Shuffle to distribute addresses with different number of transactions
    #     chunk_size = max(1, len(addresses) // self.n_chunks)
    #     chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]

    #     args_list = []
    #     for i, chunk in enumerate(chunks):
    #         accounts_chunk = {address: self.accounts[address] for address in chunk}
    #         args_list.append((chunk, accounts_chunk, self.min_block_number, self.save_every_n, self.LIMIT, i%self.n_cores+1))

    #     with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
    #         futures = [executor.submit(process_chunk_velocities, args) for args in args_list]

    #         for future in tqdm(futures, position=0):
    #             chunk_results = future.result()
    #             self.velocities.update(chunk_results)

    def save_results(self) -> None:
        """Save final results to pickle file.
        
        Saves results in the exact format: [backup_accounts, velocities, balances]
        This format is preserved for backwards compatibility.
        If split_save is enabled, results are already saved in separate files.
        """
        if self.split_save:
            # Results already saved in separate files during processing
            return
        else:
            # Save all results in single pickle file with exact format
            with open(self.output_file, 'wb') as file:
                pickle.dump([self.backup_accounts, self.velocities, self.balances], file)

    def run_analysis(self) -> None:
        """Run the complete micro-velocity analysis pipeline.
        
        This is the main entry point that orchestrates the entire analysis:
        1. Load allocated and transfer data from CSV files
        2. Calculate the analysis grid and create backup
        3. Calculate balances at each save point
        4. Calculate velocities using the matching algorithm
        5. Save results to pickle file
        
        The analysis supports both single-core and parallel execution modes.
        """
        # Step 1: Load input data
        print("Loading allocated data...", self.allocated_file)
        self.load_allocated_data()
        
        print("Loading transfer data...", self.transfers_file)
        self.load_transfer_data()
        
        # Step 2: Set up analysis parameters
        print("Computing interval of", self.save_every_n, "blocks")
        print(f"Min block number: {self.min_block_number}")
        print(f"Max block number: {self.max_block_number}")
        
        # Calculate number of save points in the analysis grid
        self.LIMIT = (self.max_block_number - self.min_block_number) // self.save_every_n + 1
        
        # Create deep copy backup before any mutations (important fix)
        self.backup_accounts = copy.deepcopy(self.accounts)
        
        print(f"Number of blocks considered: {self.LIMIT}")
        
        # Step 3: Calculate balances and velocities
        print("Calculating balances...")
        if self.n_cores == 1:
            # Single-core execution
            self.calculate_balances()
            print("Calculating velocities...")
            self.calculate_velocities()
        else:
            # Parallel execution
            self.calculate_balances_parallel()
            print("Calculating velocities...")
            self.calculate_velocities_parallel()
        
        # Step 4: Save results
        print("Saving results...")
        self.save_results()
        print("Done!")

def main():
    """Main entry point for the command-line interface.
    
    Parses command-line arguments and runs the micro-velocity analysis.
    Supports all original parameters plus new normalization options.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='Micro Velocity Analyzer with optional velocity normalization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default settings
  python micro_velocity_analyzer.py
  
  # Analysis with balance normalization
  python micro_velocity_analyzer.py --normalize_velocities by_balance
  
  # Analysis with supply normalization
  python micro_velocity_analyzer.py --normalize_velocities by_supply --total_supply 1000000
  
  # Parallel processing with 4 cores
  python micro_velocity_analyzer.py --n_cores 4 --n_chunks 8
"""
    )
    
    # Input/Output parameters
    parser.add_argument('--allocated_file', type=str, default='sampledata/sample_allocated.csv', 
                       help='Path to the allocated CSV file (initial token distributions)')
    parser.add_argument('--transfers_file', type=str, default='sampledata/sample_transfers.csv', 
                       help='Path to the transfers CSV file (token transfers between addresses)')
    parser.add_argument('--output_file', type=str, default='sampledata/general_velocities.pickle', 
                       help='Path to the output pickle file')
    
    # Analysis parameters
    parser.add_argument('--save_every_n', type=int, default=1, 
                       help='Save every Nth position of the velocity array (must be positive). '
                            'Higher values reduce memory usage but lower temporal resolution.')
    
    # Parallel processing parameters
    parser.add_argument('--n_cores', type=int, default=1, 
                       help='Number of CPU cores to use for parallel processing')
    parser.add_argument('--n_chunks', type=int, default=1, 
                       help='Number of chunks to split the data into (should be >= n_cores for efficiency)')
    parser.add_argument('--split_save', action='store_true', default=False, 
                       help='Split the save into different files instead of one large file')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Number of chunks to process in a single batch (affects memory usage)')
    
    # Normalization parameters (new feature)
    parser.add_argument('--normalize_velocities', type=str, default='none', 
                       choices=['none', 'by_balance', 'by_supply'],
                       help='Velocity normalization mode:\n'
                            '  none: Absolute velocities (default, preserves original behavior)\n'
                            '  by_balance: Normalize by account balance at each time point\n'
                            '  by_supply: Normalize by total token supply (constant)')
    parser.add_argument('--total_supply', type=float, default=None,
                       help='Total token supply for by_supply normalization '
                            '(required when normalize_velocities=by_supply)')
    
    # Parse arguments
    args = parser.parse_args()

    # Create and run analyzer
    analyzer = MicroVelocityAnalyzer(
        allocated_file=args.allocated_file,
        transfers_file=args.transfers_file,
        output_file=args.output_file,
        save_every_n=args.save_every_n,
        n_cores=args.n_cores,
        n_chunks=args.n_chunks,
        split_save=args.split_save,
        batch_size=args.batch_size,
        normalize_velocities=args.normalize_velocities,
        total_supply=args.total_supply
    )
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
