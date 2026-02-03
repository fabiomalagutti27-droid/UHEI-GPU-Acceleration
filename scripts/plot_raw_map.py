"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/plot_raw_map.py
DESCRIPTION: Raw data visualization.
Generates unfiltered scientific visualization of risk map output.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

print("[INFO] Starting raw visualization (scientific)")

# --- CONFIGURAZIONE ---
# Assicurati che queste dimensioni corrispondano a quelle definite nel C++
WIDTH = 758
HEIGHT = 502
INPUT_FILE = "output_risk_map.bin" # Il file generato da risk_map_generator.cu
OUTPUT_IMG = "Mappa_Rischio_Raw.png"

def plot_raw():
    # 1. Verifica Esistenza
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File '{INPUT_FILE}' not found.")
        print("[INFO] Run the C++ code (risk_map_generator) first.")
        return

    # 2. Caricamento Dati
    try:
        raw_data = np.fromfile(INPUT_FILE, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] File read error: {e}")
        return

    # 3. Controllo Integrità
    expected = WIDTH * HEIGHT
    if raw_data.size != expected:
        print(f"[ERROR] Dimension mismatch: found {raw_data.size} pixels, expected {expected} ({WIDTH}x{HEIGHT}).")
        return

    # 4. Reshape
    heatmap = raw_data.reshape((HEIGHT, WIDTH))

    # 5. Plotting (Inferno è percettivamente uniforme, ottimo per la scienza)
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='inferno')
    plt.colorbar(label='Indice Rischio (UHEI)')
    plt.title("Thermal Risk Map (Raw Data)\nNo filters applied")
    plt.axis('off')

    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"[INFO] Raw map saved: {OUTPUT_IMG}")
    # plt.show() # Decommentare se si esegue in ambiente grafico

if __name__ == "__main__":
    plot_raw()