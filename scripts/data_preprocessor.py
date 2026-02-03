"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/data_preprocessor.py
DESCRIPTION: Data preprocessing pipeline.
Converts GeoTIFF raster files to binary format (float32) for GPU processing.
"""

import os
import numpy as np
import rasterio
import cv2
import sys

print("[INFO] Starting data preprocessing (TIFF -> BIN)")

# --- CONFIGURAZIONE ---
# Mappa: Nome Logico -> Nome File Input
FILES = {
    "NDVI": "ndvi_mean.tif", 
    "LST": "lst_mean.tif", 
    "ALB": "albedo_mean.tif"
}

# Dimensioni di default (Fallback se manca il file master)
DEFAULT_W, DEFAULT_H = 758, 502

# --- FASE 1: DETERMINAZIONE DIMENSIONI TARGET ---
# Usiamo NDVI come "Master" per la risoluzione, perché solitamente ha la qualità migliore
target_w, target_h = DEFAULT_W, DEFAULT_H
master_file = FILES["NDVI"]

if os.path.exists(master_file):
    with rasterio.open(master_file) as src:
        target_w, target_h = src.width, src.height
    print(f"[INFO] Target dimensions (based on NDVI): {target_w} x {target_h}")
else:
    print(f"[WARN] Master file {master_file} not found. Using default: {target_w} x {target_h}")

# --- FASE 2: ELABORAZIONE E SALVATAGGIO ---
print("\n[INFO] Processing input files...")

for name, fname in FILES.items():
    if not os.path.exists(fname):
        print(f"[ERROR] Missing file {fname}. Cannot proceed for {name}.")
        continue

    try:
        with rasterio.open(fname) as src:
            # Lettura prima banda
            data = src.read(1)
            
            # Gestione NaN (Sostituiamo con 0.0 per non rompere i calcoli GPU)
            data = np.nan_to_num(data, nan=0.0)

            # Controllo Dimensioni e Resize se necessario
            current_h, current_w = data.shape
            if (current_w, current_h) != (target_w, target_h):
                print(f"[INFO] Resizing {name} ({current_w}x{current_h} -> {target_w}x{target_h})...")
                # INTER_LINEAR è ottimo per dati continui come temperatura/riflettanza
                data = cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                print(f"[INFO] {name} dimensions OK ({current_w}x{current_h}).")

            # Conversione a float32 (formato richiesto dalla GPU)
            data_float = data.astype(np.float32)

            # Definizione nome output
            # Mappiamo i nomi logici sui nomi che il C++ si aspetta
            if name == "ALB": out_name = "input_albedo.bin"
            else:             out_name = f"input_{name.lower()}.bin"

            # Salvataggio Binario (Raw)
            data_float.tofile(out_name)
            print(f"[INFO] Saved: {out_name} ({len(data_float.flatten()) * 4 / 1024 / 1024:.2f} MB)")

    except Exception as e:
        print(f"[ERROR] Critical error processing {fname}: {e}")

# --- FASE 3: REPORT FINALE PER IL C++ ---
print("\n[RESULTS] CUDA configuration summary:")
print("="*40)
print("Ensure the .cu file includes:")
print(f"#define WIDTH  {target_w}")
print(f"#define HEIGHT {target_h}")
print("="*40)