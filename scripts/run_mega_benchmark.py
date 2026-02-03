"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/run_mega_benchmark.py
DESCRIPTION: Comprehensive benchmark orchestrator.
Compiles and executes all kernel configurations with automated result collection.
"""

import os
import subprocess
import pandas as pd
import math

print("[INFO] Ultimate Benchmark: CPU vs NAIVE vs SHARED vs OPTIMIZED")
print("[INFO] Validation, Speedup, Latency, Batch Analysis")

# --- 1. CONFIGURAZIONE ---
PROJECT_DIR = "."
SRC_FILE = os.path.join("src", "benchmark_suite.cu")
BIN_DIR = "bin"
if not os.path.exists(BIN_DIR): os.makedirs(BIN_DIR)

# Rilevamento Architettura GPU
try:
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode().strip()
    if "T4" in gpu_info: arch_flag = "-arch=sm_75"
    elif "L4" in gpu_info: arch_flag = "-arch=sm_89"
    elif "A100" in gpu_info: arch_flag = "-arch=sm_80"
    else: arch_flag = ""
except: 
    arch_flag = ""
    gpu_info = "Unknown"

print(f"[INFO] GPU: {gpu_info} ({arch_flag})")
print(f"[INFO] Source: {SRC_FILE}")

# --- 2. DEFINIZIONE TEST SUITE ---
# Qui definiamo le "ricette" di compilazione
tests = [
    # Nome, BlockDim, Flag di compilazione, Descrizione
    {"name": "Naive (32x32)", "dim": 32, "flags": "-DUSE_NAIVE", "desc": "Baseline Global Mem"},
    {"name": "Shared (8x8)",   "dim": 8,  "flags": "", "desc": "Tiling, low occupancy"},
    {"name": "Shared (16x16)", "dim": 16, "flags": "", "desc": "Tiling, med occupancy"},
    {"name": "Shared (32x32)", "dim": 32, "flags": "", "desc": "Tiling, max occupancy"},
    {"name": "Optimized",      "dim": 32, "flags": "-DUSE_PINNED -DUSE_CARVEOUT --use_fast_math", "desc": "Pinned + FastMath + Carveout"}
]

results = []

# --- 3. ESECUZIONE CICLICA ---
print("\n+----------------------+----------+-----------------------------------------------+")
print("| CONFIGURATION        | STATUS   | DETAILS (Avg 5 runs)                          |")
print("+----------------------+----------+-----------------------------------------------+")

for t in tests:
    # Nome eseguibile univoco
    bin_name = os.path.join(BIN_DIR, f"bench_{t['name'].replace(' ', '_').replace('(', '').replace(')', '')}")
    
    # Comando di compilazione: Inietta le definizioni (-D) nel codice C++
    cmd_compile = f"nvcc -O3 {arch_flag} -D BLOCK_DIM={t['dim']} {t['flags']} {SRC_FILE} -o {bin_name}"
    
    try:
        # A. Compilazione
        subprocess.run(cmd_compile, shell=True, check=True)
        
        # B. Esecuzione
        res = subprocess.run(bin_name, shell=True, capture_output=True, text=True, check=True)
        
        # C. Parsing Output
        ms_cpu = ms_kernel = ms_total = ms_transfer = ms_batch = rmse = max_error = None
        kernel_type = memory_type = carveout_type = "UNKNOWN"
        status = "UNKNOWN"

        for line in res.stdout.splitlines():
            if "TEST PASSED" in line:
                status = "PASSED"
            elif "TEST FAILED" in line:
                status = "FAILED"
            if line.startswith("BENCHMARK,"):
                parts = line.split(",")
                if len(parts) >= 11:
                    kernel_type = parts[1]
                    memory_type = parts[2]
                    carveout_type = parts[3]
                    ms_cpu = float(parts[4])
                    ms_kernel = float(parts[5])
                    ms_total = float(parts[6])
                    ms_transfer = float(parts[7])
                    ms_batch = float(parts[8])
                    rmse = float(parts[9])
                    max_error = float(parts[10])

        if ms_cpu is None:
            raise ValueError("Missing BENCHMARK output line")
        
        # Calcoli derivati
        time_per_frame_batch = ms_batch / 1000.0
        speedup_kernel = ms_cpu / ms_kernel if ms_kernel > 0 else 0
        speedup_total = ms_cpu / ms_total if ms_total > 0 else 0
        transfer_overhead = ((ms_total - ms_kernel) / ms_total * 100.0) if ms_total > 0 else 0
        validity = "OK" if rmse < 1e-3 else "FAIL"
        
        # Salvataggio Dati
        results.append({
            "Config": t['name'],
            "Kernel": kernel_type,
            "Memory": memory_type,
            "Carveout": carveout_type,
            "CPU Time (ms)": ms_cpu,
            "Kernel Avg (5 runs) (ms)": ms_kernel,
            "Total Avg (5 runs) (ms)": ms_total,
            "Transfer Avg (5 runs) (ms)": ms_transfer,
            "Batch Total Avg (5 runs) (1k) ms": ms_batch,
            "Calc/Frame Avg (5 runs) (ms)": time_per_frame_batch,
            "Speedup (Kernel)": speedup_kernel,
            "Speedup (Total)": speedup_total,
            "Transfer Overhead (%)": transfer_overhead,
            "RMSE": rmse,
            "Max Error": max_error,
            "Valid": validity,
            "Status": status
        })

        print(
            f"| {t['name']:<20} | {status:<8} | Kernel: {ms_kernel:6.3f} ms | Total: {ms_total:6.3f} ms | Transfer: {ms_transfer:6.3f} ms |"
        )
        
    except Exception as e:
        print(f"| {t['name']:<20} | FAIL     | {e}")

    print("+----------------------+----------+-----------------------------------------------+")

# --- 4. REPORT E SALVATAGGIO ---
if results:
    df = pd.DataFrame(results)

    # Salvataggio CSV Finale (Fondamentale per i grafici)
    csv_path = "final_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Results saved to: '{csv_path}'")

    print("\n[RESULTS] Final Performance Table (Avg 5 runs)")
    print("+----------------------+--------+---------+----------+---------------+----------------------+----------------------+------------------------+-------------------------------+-------------------------------+------------+------------+----------+----------+--------+")
    print("| Config               | Kernel | Memory  | Carveout | CPU Time (ms) | Kernel Time (ms)     | Total Time (ms)      | Transfer Overhead (%) | Batch Total Avg (5 runs) ms  | Calc/Frame Avg (5 runs) ms    | Speedup-K  | Speedup-T  | RMSE     | Max Err  | Valid  |")
    print("+----------------------+--------+---------+----------+---------------+----------------------+----------------------+------------------------+-------------------------------+-------------------------------+------------+------------+----------+----------+--------+")
    for _, row in df.iterrows():
        print(
            f"| {row['Config']:<20} | {row['Kernel']:<6} | {row['Memory']:<7} | {row['Carveout']:<8} | "
            f"{row['CPU Time (ms)']:<13.4f} | {row['Kernel Avg (5 runs) (ms)']:<20.4f} | {row['Total Avg (5 runs) (ms)']:<20.4f} | "
            f"{row['Transfer Overhead (%)']:<22.2f} | {row['Batch Total Avg (5 runs) (1k) ms']:<29.1f} | {row['Calc/Frame Avg (5 runs) (ms)']:<29.4f} | "
            f"{row['Speedup (Kernel)']:<10.2f} | {row['Speedup (Total)']:<10.2f} | {row['RMSE']:<8.2e} | {row['Max Error']:<8.2e} | {row['Valid']:<6} |"
        )
    print("+----------------------+--------+---------+----------+---------------+----------------------+----------------------+------------------------+-------------------------------+-------------------------------+------------+------------+----------+----------+--------+")

    # Analisi Automatica "Intelligente"
    best_total = df.loc[df['Total Avg (5 runs) (ms)'].idxmin()]
    best_kernel = df.loc[df['Kernel Avg (5 runs) (ms)'].idxmin()]
    best_batch = df.loc[df['Batch Total Avg (5 runs) (1k) ms'].idxmin()]
    
    print("\n[RESULTS] Champions (Avg 5 runs):")
    print(f"  Lowest Total Latency (User Time):    {best_total['Config']} - {best_total['Total Avg (5 runs) (ms)']:.3f} ms")
    print(f"  Lowest Kernel Time (Compute Only):   {best_kernel['Config']} - {best_kernel['Kernel Avg (5 runs) (ms)']:.3f} ms")
    print(f"  Highest Batch Throughput:            {best_batch['Config']} - {best_batch['Batch Total Avg (5 runs) (1k) ms']:.1f} ms (1000 frames)")

else:
    print("\n[ERROR] No results collected. Check compilation errors.")