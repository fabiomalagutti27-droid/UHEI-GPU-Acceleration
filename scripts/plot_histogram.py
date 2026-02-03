"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/plot_histogram.py
DESCRIPTION: Statistical distribution analysis.
Generates histogram of risk index values with statistical metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

print("[INFO] Starting statistical analysis (histogram)")

# --- CONFIGURAZIONE ---
WIDTH = 758
HEIGHT = 502
# Cerchiamo il file generato dal codice C++ principale
INPUT_FILE = "output_risk_map.bin"
OUTPUT_IMG = "Grafico_Distribuzione_Rischio.png"

def plot_distribution():
    # 1. Controllo Esistenza File
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File '{INPUT_FILE}' not found.")
        print("[INFO] Run 'risk_map_generator' first.")
        return

    try:
        # 2. Caricamento Dati
        data = np.fromfile(INPUT_FILE, dtype=np.float32)

        # Check dimensioni
        expected = WIDTH * HEIGHT
        if data.size != expected:
            print(f"[WARN] The file contains {data.size} pixels, expected {expected}.")

        # 3. Filtro Dati Validi
        # Ignoriamo i valori < 0.001 (che sono i bordi neri/padding o aree senza dati)
        # Questo è CRUCIALE: se includi gli zeri, l'istogramma viene schiacciato da una colonna enorme a sinistra.
        valid_data = data[data > 0.001]

        if valid_data.size == 0:
            print("[WARN] No valid data found (empty image?).")
            return

        print(f"[INFO] Analysis on {valid_data.size} valid pixels (excluding borders/padding).")
        
        # Statistiche Base
        mean_val = np.mean(valid_data)
        median_val = np.median(valid_data)
        std_val = np.std(valid_data)

        # 4. PLOT ISTOGRAMMA
        plt.figure(figsize=(10, 6))

        # bins=100: ottima granularità per vedere la curva gaussiana
        # density=True: normalizza l'area a 1 (opzionale, qui usiamo frequenza assoluta)
        n, bins, patches = plt.hist(valid_data, bins=100, color='#ff9900', alpha=0.7, edgecolor='grey')

        # Linee Statistiche
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Media: {mean_val:.3f}')
        plt.axvline(median_val, color='blue', linestyle='dotted', linewidth=2, label=f'Mediana: {median_val:.3f}')
        
        # Deviazione Standard (opzionale, ma fa scena scientifica)
        plt.axvspan(mean_val - std_val, mean_val + std_val, color='red', alpha=0.1, label='±1 Std Dev')

        plt.title('Thermal Risk Distribution (UHEI) - Bologna', fontsize=14, fontweight='bold')
        plt.xlabel('Indice di Rischio', fontsize=12)
        plt.ylabel('Numero di Pixel (Frequenza)', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Salvataggio
        plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
        print(f"[INFO] Histogram saved: {OUTPUT_IMG}")
        
        # Print Statistiche a console (utile per il report testuale)
        print("\n[RESULTS] Statistical Summary:")
        print(f"   Mean:   {mean_val:.4f}")
        print(f"   Median: {median_val:.4f}")
        print(f"   Std Dev: {std_val:.4f}")
        print(f"   Min/Max: {np.min(valid_data):.4f} / {np.max(valid_data):.4f}")

    except Exception as e:
        print(f"[ERROR] Processing error: {e}")

if __name__ == "__main__":
    plot_distribution()