"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/run_phase5_benchmark.py
DESCRIPTION: Phase 5 benchmark - Functional verification.
Tests multiple block sizes with CPU baseline comparison and RMSE validation.
"""

import os
import subprocess

print("[INFO] Phase 5: Functional Verification and Baseline")

SRC_FILE = os.path.join("src", "risk_map_generator.cu")
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

print(f"[INFO] GPU Target: {gpu_info}")

dims = [8, 16, 32]
results = []

print("\n[INFO] Running preliminary benchmark...")
for d in dims:
    exe_name = os.path.join(BIN_DIR, f"bench_{d}")
    cmd_compile = f"nvcc -O3 {arch_flag} -D BLOCK_DIM={d} {SRC_FILE} -o {exe_name}"
    
    # Compilazione (Silenziosa se ok)
    ret = subprocess.run(cmd_compile, shell=True, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] Compile Error {d}x{d}: {ret.stderr}")
        continue
        
    # Esecuzione
    ret_run = subprocess.run(exe_name, shell=True, capture_output=True, text=True)

    # Parsing
    status = "UNKNOWN"
    csv_parts = None
    for line in ret_run.stdout.splitlines():
        if "TEST PASSED" in line:
            status = "PASSED"
        elif "TEST FAILED" in line:
            status = "FAILED"
        if line.startswith("CSV_DATA"):
            csv_parts = line.split(",")

    if csv_parts and len(csv_parts) >= 6:
        results.append({
            "dim": int(csv_parts[1]),
            "cpu": float(csv_parts[2]),
            "gpu": float(csv_parts[3]),
            "rmse": float(csv_parts[4]),
            "max_err": float(csv_parts[5]),
            "status": status
        })

# Stampa Tabella Tecnica
print("\n[RESULTS] Phase 5 Summary")
print("+----------+------------+------------+----------+--------------+--------------+----------+")
print(f"| {'BLOCK':<8} | {'CPU (ms)':<10} | {'GPU (ms)':<10} | {'SPEEDUP':<8} | {'RMSE':<12} | {'MAX_ERR':<12} | {'STATUS':<8} |")
print("+----------+------------+------------+----------+--------------+--------------+----------+")
for r in results:
    speedup = r['cpu'] / r['gpu'] if r['gpu'] > 0 else 0
    print(
        f"| {r['dim']}x{r['dim']:<5} | {r['cpu']:<10.3f} | {r['gpu']:<10.3f} | {speedup:<6.1f}x | {r['rmse']:<12.2e} | {r['max_err']:<12.2e} | {r['status']:<8} |"
    )
print("+----------+------------+------------+----------+--------------+--------------+----------+")