#!/usr/bin/env python3
"""
Interactive explorer for the Zarr archive.

Usage:
    uv run python explore_zarr.py [path_to_zarr]
    
Examples:
    # Browse the archive interactively
    uv run python explore_zarr.py
    
    # List all experiments
    uv run python explore_zarr.py --list
    
    # Show details for a specific experiment
    uv run python explore_zarr.py --experiment SEN07/lclog75/1p2mpm/6lpm/PIV01
    
    # Print tree structure
    uv run python explore_zarr.py --tree
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zarr.storage


def print_tree(group: zarr.Group, prefix: str = "", max_depth: int = 5, depth: int = 0) -> None:
    """Print a tree structure of the Zarr group."""
    if depth >= max_depth:
        return
    
    items = sorted(group.keys())
    for i, key in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{key}")
        
        item = group[key]
        if hasattr(item, "keys"):
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(item, next_prefix, max_depth, depth + 1)


def list_experiments(root: zarr.Group) -> None:
    """List all experiments in the archive."""
    runs = root.get("runs")
    if not runs:
        print("No 'runs' group found in archive.")
        return
    
    experiments = []
    
    def collect_experiments(group: zarr.Group, path: str = "") -> None:
        for key in sorted(group.keys()):
            item = group[key]
            current_path = f"{path}/{key}" if path else key
            if hasattr(item, "keys"):
                if "piv" in item:
                    # This is an experiment
                    attrs = item.attrs
                    experiments.append({
                        "path": current_path,
                        "sen": attrs.get("sen", "?"),
                        "variant": attrs.get("variant", "?"),
                        "speed": attrs.get("strand_speed_slug", "?"),
                        "gas": attrs.get("gas_flow_lpm", "?"),
                        "run": attrs.get("piv_run", "?"),
                        "frames": attrs.get("n_frames", "?"),
                    })
                else:
                    collect_experiments(item, current_path)
    
    collect_experiments(runs)
    
    print(f"Found {len(experiments)} experiments:\n")
    print(f"{'Path':<50} {'SEN':<8} {'Variant':<10} {'Speed':<8} {'Gas':<6} {'Run':<6} {'Frames':<8}")
    print("-" * 100)
    for exp in experiments:
        print(
            f"{exp['path']:<50} {exp['sen']:<8} {exp['variant']:<10} "
            f"{exp['speed']:<8} {exp['gas']:<6}LPM PIV{exp['run']:02d} {exp['frames']:<8}"
        )


def show_experiment(root: zarr.Group, experiment_path: str) -> None:
    """Show detailed information about a specific experiment."""
    path_parts = [p for p in experiment_path.split("/") if p]
    
    if path_parts[0] != "runs":
        path_parts.insert(0, "runs")
    
    try:
        current = root
        for part in path_parts:
            current = current[part]
        
        exp = current
        
        print(f"\n=== Experiment: {'/'.join(path_parts)} ===\n")
        
        # Show attributes
        print("Attributes:")
        for key in sorted(exp.attrs.keys()):
            value = exp.attrs[key]
            if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
        
        # Show groups and arrays
        print("\nGroups and Arrays:")
        for key in sorted(exp.keys()):
            item = exp[key]
            if hasattr(item, "shape"):
                # It's an array
                print(f"  {key}/ (array): shape={item.shape}, dtype={item.dtype}, chunks={item.chunks}")
            elif hasattr(item, "keys"):
                # It's a group
                print(f"  {key}/ (group):")
                for subkey in sorted(item.keys()):
                    subitem = item[subkey]
                    if hasattr(subitem, "shape"):
                        print(f"    {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
                    else:
                        print(f"    {subkey}/ (group)")
        
        # Show sample data
        print("\nSample Data:")
        if "piv" in exp:
            piv = exp["piv"]
            if "u" in piv:
                u = piv["u"]
                print(f"  PIV u: shape={u.shape}, range=[{u[:].min():.4f}, {u[:].max():.4f}]")
            if "time_s" in piv:
                time = piv["time_s"]
                print(f"  PIV time_s: shape={time.shape}, range=[{time[:].min():.4f}, {time[:].max():.4f}]")
        
        if "sensor" in exp:
            sensor = exp["sensor"]
            if "table" in sensor:
                table = sensor["table"]
                print(f"  Sensor table: shape={table.shape}, dtype={table.dtype}")
            if "columns" in sensor.attrs:
                cols = sensor.attrs["columns"]
                print(f"  Sensor columns: {len(cols)} columns")
                print(f"    First 5: {cols[:5]}")
        
    except KeyError as e:
        print(f"Error: Experiment path not found: {experiment_path}")
        print(f"Missing key: {e}")
        sys.exit(1)


def interactive_browse(root: zarr.Group) -> None:
    """Interactive browser for the archive."""
    print("\n=== Zarr Archive Explorer ===\n")
    print("Commands:")
    print("  tree [depth]  - Show tree structure (default depth: 5)")
    print("  list          - List all experiments")
    print("  show <path>   - Show experiment details")
    print("  cd <path>     - Navigate to group")
    print("  ls            - List current group")
    print("  pwd           - Show current path")
    print("  quit/exit     - Exit\n")
    
    current = root
    current_path = []
    
    while True:
        try:
            cmd = input(f"{'/'.join(current_path) or '/'}> ").strip().split()
            if not cmd:
                continue
            
            command = cmd[0].lower()
            
            if command in ("quit", "exit", "q"):
                break
            elif command == "tree":
                depth = int(cmd[1]) if len(cmd) > 1 else 5
                print_tree(current, max_depth=depth)
            elif command == "list":
                list_experiments(root)
            elif command == "show":
                if len(cmd) < 2:
                    print("Usage: show <experiment_path>")
                    continue
                show_experiment(root, cmd[1])
            elif command == "cd":
                if len(cmd) < 2:
                    current = root
                    current_path = []
                    print("Reset to root")
                    continue
                
                target = cmd[1]
                if target == "..":
                    if current_path:
                        current_path.pop()
                        current = root
                        for part in current_path:
                            current = current[part]
                elif target == "/":
                    current = root
                    current_path = []
                else:
                    if target in current:
                        item = current[target]
                        if hasattr(item, "keys"):
                            current = item
                            current_path.append(target)
                        else:
                            print(f"{target} is not a group")
                    else:
                        print(f"Key '{target}' not found")
            elif command == "ls":
                items = sorted(current.keys())
                for key in items:
                    item = current[key]
                    if hasattr(item, "shape"):
                        print(f"  {key} (array): {item.shape} {item.dtype}")
                    else:
                        print(f"  {key}/ (group)")
            elif command == "pwd":
                print("/" + "/".join(current_path))
            else:
                print(f"Unknown command: {command}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore Zarr archive interactively or with commands"
    )
    parser.add_argument(
        "archive_path",
        nargs="?",
        default="data/processed/all_experiments.zarr",
        help="Path to Zarr archive directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print tree structure",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Show details for specific experiment (e.g., SEN07/lclog75/1p2mpm/6lpm/PIV01)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Maximum depth for tree view",
    )
    
    args = parser.parse_args()
    
    archive_path = Path(args.archive_path)
    if not archive_path.exists():
        print(f"Error: Archive not found at {archive_path}")
        sys.exit(1)
    
    store = zarr.storage.LocalStore(str(archive_path))
    root = zarr.group(store=store)
    
    if args.list:
        list_experiments(root)
    elif args.tree:
        print_tree(root, max_depth=args.depth)
    elif args.experiment:
        show_experiment(root, args.experiment)
    else:
        interactive_browse(root)


if __name__ == "__main__":
    main()
