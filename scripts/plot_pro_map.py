"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/plot_pro_map.py
DESCRIPTION: High-contrast risk map visualization.
Generates publication-quality maps with dynamic contrast stretching.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

print("[INFO] Starting professional visualization (high contrast)")

# --- CONFIGURAZIONE ---
WIDTH = 758
HEIGHT = 502
INPUT_FILE = "output_risk_map.bin"
OUTPUT_IMG = "Mappa_Rischio_Pro.png"

def plot_pro():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Missing '{INPUT_FILE}'.")
        return

    # 1. Caricamento
    raw_data = np.fromfile(INPUT_FILE, dtype=np.float32)
    if raw_data.size != WIDTH * HEIGHT:
        print(f"[ERROR] Invalid dimensions ({raw_data.size} pixels).")
        return

    heatmap = raw_data.reshape((HEIGHT, WIDTH))

    # 2. Calcolo Statistiche per Contrasto
    # Ignoriamo i valori esattamente a 0 (bordi neri/padding) per il calcolo dei percentili
    valid_pixels = heatmap[heatmap > 0.00001]
    
    if valid_pixels.size == 0:
        print("[WARN] Empty image or all zeros.")
        vmin, vmax = 0, 1
    else:
        # Tagliamo il 2% inferiore e superiore (Outlier Removal)
        vmin = np.percentile(valid_pixels, 2)
        vmax = np.percentile(valid_pixels, 98)
        
        print("[INFO] Dynamic contrast statistics:")
        print(f"   Min/Max (real): {np.min(valid_pixels):.4f} / {np.max(valid_pixels):.4f}")
        print(f"   Min/Max (2-98%): {vmin:.4f} / {vmax:.4f}")

    # 3. Plotting Avanzato
    plt.figure(figsize=(12, 10))
    
    # RdYlBu_r: Red=Caldo (Alto Rischio), Blue=Freddo (Basso Rischio)
    # vmin/vmax forzano il contrasto sulla parte interessante dei dati
    img = plt.imshow(heatmap, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(img, label='Indice Rischio Termico (Ottimizzato)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    plt.title("Thermal Risk Analysis - Bologna\n(Contrast Stretching 2%-98%)", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"[INFO] Pro map saved: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_pro()