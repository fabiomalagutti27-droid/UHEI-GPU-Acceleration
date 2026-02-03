"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/run_tuning_extreme.py
DESCRIPTION: Exploratory instruction-level optimization test.
Compares standard vs extreme-optimized Naive kernel to quantify memory-bound constraints.
"""

import os
import subprocess

print("[INFO] Deathmatch: Naive Standard vs Naive Extreme (Batch 1000)")

# --- 1. CONFIGURAZIONE ---
PROJECT_DIR = "."
SRC_FILE = os.path.join("src", "kernel_tuning_extreme.cu")
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

# --- 2. COMPILAZIONE ---
exe_name = os.path.join(BIN_DIR, "tuning_extreme_battle")
print("\n[INFO] Compiling battle...")

# Aggiungiamo --use_fast_math per abilitare le ottimizzazioni hardware
cmd_compile = f"nvcc -O3 {arch_flag} --use_fast_math {SRC_FILE} -o {exe_name}"
try:
    subprocess.run(cmd_compile, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Compilation failed: {e}")
    exit(1)

# --- 3. ESECUZIONE ---
print("[INFO] Running battle (1000 iterations)...")
try:
    res = subprocess.run(exe_name, shell=True, capture_output=True, text=True, check=True)
    std_ms = None
    ext_ms = None
    for line in res.stdout.splitlines():
        if "Kernel Standard:" in line:
            std_ms = float(line.split(":")[1].strip().split()[0])
        elif "Kernel Extreme:" in line:
            ext_ms = float(line.split(":")[1].strip().split()[0])

    print("+-----------------+----------+")
    print("| Kernel          | Time (ms) |")
    print("+-----------------+----------+")
    if std_ms is not None:
        print(f"| Standard        | {std_ms:<8.3f} |")
    if ext_ms is not None:
        print(f"| Extreme         | {ext_ms:<8.3f} |")
    print("+-----------------+----------+")

    if std_ms is not None and ext_ms is not None and std_ms > 0:
        delta = std_ms - ext_ms
        pct = (delta / std_ms) * 100.0
        sign = "-" if delta > 0 else "+"
        print(f"[RESULTS] Delta: {sign}{abs(delta):.3f} ms ({sign}{abs(pct):.2f}%)")

    print("[NOTE] This stencil is memory-bound; low-level optimizations typically yield limited gains.")
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Execution failed: {e}")