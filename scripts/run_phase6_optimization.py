"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/run_phase6_optimization.py
DESCRIPTION: Phase 6 benchmark - System-level optimization analysis.
Measures bandwidth and throughput with Pinned Memory configuration.
"""

import os
import subprocess

print("[INFO] Phase 6: Latency and Bandwidth Analysis (32x32 vs 16x16)")

SRC_FILE = os.path.join("src", "gpu_optimized_test.cu")
BIN_DIR = "bin"
if not os.path.exists(BIN_DIR): os.makedirs(BIN_DIR)

# Configurazione Immagine per calcolo Banda
# 3 Input float + 1 Output float = 16 bytes per pixel
BYTES_PER_PIXEL = 16 
TOTAL_PIXELS = 758 * 502
TOTAL_GB = (TOTAL_PIXELS * BYTES_PER_PIXEL) / 1e9

# Architettura
try:
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode().strip()
    if "T4" in gpu_info: arch_flag = "-arch=sm_75"
    elif "L4" in gpu_info: arch_flag = "-arch=sm_89"
    elif "A100" in gpu_info: arch_flag = "-arch=sm_80"
    else: arch_flag = ""
except: arch_flag = ""

configs = [32, 16]

print("+----------+------------+------------------+----------------------+------------+")
print(f"| {'CONFIG':<8} | {'TIME (ms)':<10} | {'BANDWIDTH (GB/s)':<16} | {'THROUGHPUT (GPix/s)':<20} | {'RMSE':<10} |")
print("+----------+------------+------------------+----------------------+------------+")

for block_dim in configs:
    exe_name = os.path.join(BIN_DIR, f"opt_test_{block_dim}")
    cmd_compile = f"nvcc -O3 {arch_flag} --use_fast_math -D BLOCK_DIM={block_dim} {SRC_FILE} -o {exe_name}"
    
    subprocess.run(cmd_compile, shell=True, check=True)
    ret = subprocess.run(exe_name, shell=True, capture_output=True, text=True)
    
    for line in ret.stdout.splitlines():
        if line.startswith("TEST_RESULT"):
            # Format: TEST_RESULT, DIM, MS, RMSE
            parts = line.split(",")
            ms = float(parts[2])
            rmse = float(parts[3])
            
            # Calcoli Fisici
            bw = 0
            gpix = 0
            if ms > 0:
                bw = TOTAL_GB / (ms / 1000.0) # GB/s
                gpix = (TOTAL_PIXELS / 1e9) / (ms / 1000.0) # GPixel/s

            print(f"| {parts[1]}x{parts[1]:<5} | {ms:<10.5f} | {bw:<16.2f} | {gpix:<20.2f} | {rmse:<10.2e} |")
print("+----------+------------+------------------+----------------------+------------+")