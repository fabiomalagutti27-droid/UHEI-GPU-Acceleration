import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/plot_performance_charts.py
DESCRIPTION: Performance visualization suite.
Generates charts for sensitivity analysis, throughput, and speedup comparison.
"""

print("[INFO] Generating performance charts")

# --- 1. CONFIGURAZIONE ---
# Dimensioni di default per eventuali calcoli di banda (non critico per questi grafici, ma buona pratica)
WIDTH_IMG = 758
HEIGHT_IMG = 502
CSV_PATH = "final_results.csv"

def generate_charts():
    # --- 2. CARICAMENTO DATI ---
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Missing '{CSV_PATH}'.")
        print("[INFO] Run 'run_mega_benchmark.py' first to generate data.")
        return

    df = pd.read_csv(CSV_PATH)
    print("[INFO] Data loaded successfully.")

    # =======================================================
    # GRAFICO 1: SENSITIVITÀ AL BLOCK SIZE (Line Plot)
    # =======================================================
    # Obiettivo: Mostrare l'impatto della dimensione del blocco sulla Latenza (Single Shot)
    df_sens = df[df['Config'].str.contains("Shared")].copy()

    if not df_sens.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(df_sens['Config'], df_sens['Total Avg (5 runs) (ms)'], 
                 marker='o', color='purple', linewidth=2, markersize=8, linestyle='-')
        
        plt.title('Sensitivity Analysis: Block Size vs Total Latency', fontsize=12, fontweight='bold')
        plt.ylabel('Total Avg (5 runs) (ms) - Lower is Better')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Annotazioni
        for i, val in enumerate(df_sens['Total Avg (5 runs) (ms)']):
            plt.text(i, val + 0.005, f"{val:.3f}", ha='center', fontweight='bold', color='purple')

        plt.tight_layout()
        plt.savefig("Chart_1_Sensitivity.png", dpi=300)
        plt.close() # Chiude la figura per liberare memoria
        print("[INFO] Generated: Chart_1_Sensitivity.png")
        

    # =======================================================
    # GRAFICO 2: LA MARATONA - THROUGHPUT (FPS su 1000 Frame)
    # =======================================================
    # Obiettivo: Mostrare la potenza bruta nel batch processing
    df_gpu = df[df['Config'].str.contains("Naive") | df['Config'].str.contains("Shared") | df['Config'].str.contains("Optimized")].copy()
    
    if not df_gpu.empty:
        # Calcolo FPS = 1000 frames / (Tempo Totale Batch ms / 1000)
        df_gpu['FPS'] = 1000.0 / (df_gpu['Batch Total Avg (5 runs) (1k) ms'] / 1000.0)
        
        # Colori distintivi
        colors = ['#1f77b4' if 'Naive' in x else '#2ca02c' if 'Optimized' in x else '#ff7f0e' for x in df_gpu['Config']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_gpu['Config'], df_gpu['FPS'], color=colors, edgecolor='black', alpha=0.8)
        
        plt.title('Throughput Test: Sustained Performance (Batch 1000)', fontsize=14, fontweight='bold')
        plt.ylabel('Frames Per Second (FPS) - Higher is Better', fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Annotazioni FPS
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height):,} FPS',
                     ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig("Chart_2_Throughput_FPS.png", dpi=300)
        plt.close()
        print("[INFO] Generated: Chart_2_Throughput_FPS.png")
        

    # =======================================================
    # GRAFICO 3: TRADE-OFF (LATENZA vs THROUGHPUT)
    # =======================================================
    # Obiettivo: Scatter plot per visualizzare il compromesso architettonico
    plt.figure(figsize=(9, 6))
    
    for i, row in df_gpu.iterrows():
        plt.scatter(row['Total Avg (5 runs) (ms)'], row['Batch Total Avg (5 runs) (1k) ms'], s=150, label=row['Config'])
        plt.text(row['Total Avg (5 runs) (ms)'], row['Batch Total Avg (5 runs) (1k) ms'] + 1, 
                 row['Config'].split(" ")[0], fontsize=9, ha='center')

    plt.title('Architecture Trade-off: Total Latency vs Throughput', fontsize=14, fontweight='bold')
    plt.xlabel('Total Avg (5 runs) (ms) [Lower is Better]')
    plt.ylabel('Batch Total Avg (5 runs) (1k) ms [Lower is Better]')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Chart_3_Tradeoff.png", dpi=300)
    plt.close()
    print("[INFO] Generated: Chart_3_Tradeoff.png")
    

    # =======================================================
    # GRAFICO 4: IL CONFRONTO FINALE (CPU vs GPU - Scala Log)
    # =======================================================
    # Obiettivo: Mostrare lo speedup massiccio rispetto alla CPU
    targets = ["Naive (32x32)", "Shared (32x32)", "Optimized"]
    df_final = df[df['Config'].isin(targets)].copy()
    
    if not df_final.empty:
        cpu_time = df['CPU Time (ms)'].iloc[0]
        
        # Preparazione dati
        labels = ['CPU'] + df_final['Config'].tolist()
        times = [cpu_time] + df_final['Total Avg (5 runs) (ms)'].tolist()
        
        colors_fin = ['gray', '#1f77b4', '#ff7f0e', '#d62728']

        plt.figure(figsize=(10, 6))
        bars_fin = plt.bar(labels, times, color=colors_fin, edgecolor='black')
        
        plt.yscale('log') # FONDAMENTALE: Scala logaritmica per vedere CPU e GPU insieme
        plt.title('CPU vs GPU: Total Time Comparison (Log Scale)', fontsize=14, fontweight='bold')
        plt.ylabel('Execution Time (ms) - Log Scale')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Annotazioni Speedup
        for i, bar in enumerate(bars_fin):
            h = bar.get_height()
            if i == 0: # CPU
                plt.text(bar.get_x() + bar.get_width()/2, h * 1.1, f"{h:.1f} ms", ha='center')
            else: # GPU
                speedup = cpu_time / h
                plt.text(bar.get_x() + bar.get_width()/2, h * 1.2, f"{speedup:.1f}x", 
                         ha='center', fontweight='bold', color='black')

        plt.tight_layout()
        plt.savefig("Chart_4_CPU_vs_GPU.png", dpi=300)
        plt.close()
        print("[INFO] Generated: Chart_4_CPU_vs_GPU.png")

    # =======================================================
    # GRAFICO 5: IMPACT OF DATA TRANSFER (STACKED BAR)
    # =======================================================
    # Obiettivo: Evidenziare la quota di trasferimento vs calcolo
    targets_transfer = ["Naive (32x32)", "Shared (32x32)", "Optimized"]
    df_transfer = df[df['Config'].isin(targets_transfer)].copy()

    if not df_transfer.empty:
        df_transfer = df_transfer.set_index('Config').reindex(targets_transfer)
        kernel_times = df_transfer['Kernel Avg (5 runs) (ms)'].values
        total_times = df_transfer['Total Avg (5 runs) (ms)'].values
        transfer_times = np.maximum(total_times - kernel_times, 0.0)

        x = np.arange(len(targets_transfer))
        width = 0.6

        plt.figure(figsize=(9, 6))
        plt.bar(x, kernel_times, width, label='Kernel Time', color='#1f4e79')
        plt.bar(x, transfer_times, width, bottom=kernel_times, label='Transfer Time', color='#9ecae1')

        plt.title('Impact of Data Transfer: Compute vs I/O Bound', fontsize=14, fontweight='bold')
        plt.ylabel('Time (ms) - Lower is Better')
        plt.xticks(x, targets_transfer, rotation=10)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend()

        # Annotazioni Total Time
        for i, total in enumerate(total_times):
            plt.text(x[i], total + 0.002, f"{total:.3f} ms", ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig("Impact_of_Data_Transfer.png", dpi=300)
        plt.close()
        print("[INFO] Generated: Impact_of_Data_Transfer.png")


if __name__ == "__main__":
    generate_charts()