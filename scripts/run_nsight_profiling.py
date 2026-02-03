"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/run_nsight_profiling.py
DESCRIPTION: Nsight Compute profiling automation.
Generates detailed micro-architectural reports for kernel analysis.
"""

import os
import subprocess
import shutil

print("[INFO] Phase 8: Nsight Compute Report Generation")
print("[INFO] Memory and Compute Throughput Analysis")

# --- 1. CONFIGURAZIONE ---
PROJECT_DIR = "."
SRC_FILE = os.path.join("src", "profiling_harness.cu")
BIN_DIR = "bin"
REPORT_DIR = "reports"

if not os.path.exists(BIN_DIR): os.makedirs(BIN_DIR)
if not os.path.exists(REPORT_DIR): os.makedirs(REPORT_DIR)

# Rilevamento Architettura
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

# --- 2. COMPILAZIONE ---
print("\n[INFO] Compiling binaries for profiling...")
bin_std = os.path.join(BIN_DIR, "profile_std")
bin_fast = os.path.join(BIN_DIR, "profile_fast")

# A. Standard Build (Per Naive e Shared Base)
cmd_std = f"nvcc -O3 {arch_flag} {SRC_FILE} -o {bin_std}"
subprocess.run(cmd_std, shell=True, check=True)

# B. FastMath Build (Per Optimized)
cmd_fast = f"nvcc -O3 {arch_flag} --use_fast_math {SRC_FILE} -o {bin_fast}"
subprocess.run(cmd_fast, shell=True, check=True)

# --- 3. ESECUZIONE NSIGHT COMPUTE ---
ncu_cmd = "ncu"
 # If we are on Colab/Linux server, check the path
if os.path.exists("/usr/local/cuda/bin/ncu"): 
    ncu_cmd = "/usr/local/cuda/bin/ncu"

# Funzione Helper
def run_profile(label, bin_path, mode_arg, output_name):
    output_path = os.path.join(REPORT_DIR, output_name)
    print(f"\n[INFO] Profiling: {label}...")
    
    # --set full: Raccoglie tutte le metriche (Memory, Compute, Occupancy, Source)
    # --force-overwrite: Sovrascrive vecchi report
    # mode_arg: 0=Naive, 1=Shared
    cmd = f"{ncu_cmd} --set full --force-overwrite -o {output_path} {bin_path} {mode_arg}"
    
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[RESULTS] Report saved to: {output_path}.ncu-rep")
        else:
            print(f"[ERROR] Profiling error: {res.stderr}")
    except FileNotFoundError:
        print("[ERROR] Nsight Compute (ncu) not found in the system.")

# Esecuzione dei 3 Scenari
# 1. NAIVE (Usa binario standard, mode 0)
run_profile("NAIVE (Global Memory)", bin_std, "0", "report_01_naive")

# 2. SHARED BASE (Usa binario standard, mode 1)
run_profile("SHARED (Standard)", bin_std, "1", "report_02_shared")

# 3. OPTIMIZED (Usa binario fastmath, mode 1)
run_profile("SHARED (Optimized)", bin_fast, "1", "report_03_optimized")

# --- 4. ZIPPING ---
print("\n[INFO] Archiving reports...")
zip_name = "Nsight_Reports"
shutil.make_archive(zip_name, 'zip', REPORT_DIR)
print(f"[INFO] Download '{zip_name}.zip' and open it with Nsight Compute GUI.")