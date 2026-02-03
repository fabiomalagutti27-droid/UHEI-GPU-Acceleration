# Accelerazione GPU per l'Analisi del Rischio Ambientale

## 1. Titolo del Progetto

Accelerazione GPU per l'Analisi del Rischio Ambientale (Heat & Flood).

Descrizione: Implementazione parallela in **CUDA** C++ di un algoritmo di stenciling per il calcolo dell'indice di rischio UHEI (Urban Heat Exposure Index).

## 2. Descrizione del Dataset (Progetto TALEA)

I dati derivano dal progetto open source **TALEA** (Surface Urban Heat Islands Analysis - Bologna). Fonte ufficiale: https://github.com/TALEA-platform/suhi/tree/main/.

Dettaglio canali elaborati:

- LST (Land Surface Temperature): temperatura superficiale.
- NDVI: indice di vegetazione.
- Albedo: riflettività della superficie.

Algoritmo: Il software applica una convoluzione (stencil operation) sui tre canali per calcolare l'indice composito UHEI per ogni cella della griglia.

## 3. Struttura della Repository e Istruzioni d'Uso

Organizzazione della cartella data/:

- [data/final_gpu_v5.ipynb](data/final_gpu_v5.ipynb): Jupyter Notebook principale predisposto per l'ambiente **Google Colab**.
- [data/progetto.zip](data/progetto.zip): archivio compresso contenente l'intero codice sorgente C++ (src/) e gli script **Python** (scripts/).
- [data/test_1.zip](data/test_1.zip): archivio compresso contenente il dataset raster e vettoriale.
- [data/Mappa_Bologna_Completa.html](data/Mappa_Bologna_Completa.html): mappa interattiva dinamica generata dallo script `generate_web_map.py`, che sovrappone l'indice di rischio UHEI calcolato sulla GPU con i confini amministrativi di Bologna.

Workflow di esecuzione:

- Il notebook [data/final_gpu_v5.ipynb](data/final_gpu_v5.ipynb) deve essere aperto in **Google Colab**.
- È necessario caricare nell'ambiente di runtime i due archivi [data/progetto.zip](data/progetto.zip) e [data/test_1.zip](data/test_1.zip) mantenendoli nel loro formato compresso.
- L'esecuzione della cella di inizializzazione (Fase 0) gestirà automaticamente la decompressione, l'installazione delle dipendenze e la configurazione della gerarchia delle cartelle.

## 4. Struttura del Codice Sorgente (src/)

Elenco e descrizione dei file presenti in src/:

- **benchmark_suite.cu**: suite di test principale. Implementa il doppio timer per separare il tempo di esecuzione del kernel dal tempo totale di sistema (incluso trasferimento PCIe).
- **risk_map_generator.cu**: modulo di produzione per la generazione delle mappe di rischio e la validazione numerica (RMSE) rispetto alla baseline CPU.
- **gpu_optimized_test.cu**: modulo dedicato all'analisi della latenza di trasferimento e della banda passante, utilizzato per il confronto diretto tra Pageable Memory e Pinned Memory.
- **kernel_tuning_extreme.cu**: test esplorativo per valutare l'efficacia di ottimizzazioni a livello di istruzioni (`__restrict__`, `__launch_bounds__`, `#pragma unroll`) su algoritmi memory-bound. Conferma che il collo di bottiglia è la banda della memoria, non il throughput computazionale.
- **profiling_harness.cu**: wrapper minimale per l'analisi micro-architetturale tramite NVIDIA Nsight Compute.

## 5. Automazione e Visualizzazione (scripts/)

Toolchain di supporto nella cartella scripts/:

- **data_preprocessor.py**: conversione dei dati grezzi GeoTIFF/GeoJSON in formato binario raw (float32).
- **run_*.py**: script di orchestrazione per la compilazione (nvcc) e l'esecuzione dei vari moduli C++.
- **plot_*.py**: set di script per la generazione di grafici prestazionali (istogrammi, roofline model, analisi di sensitività).
- **generate_web_map.py**: generazione della visualizzazione geospaziale interattiva su base Folium.

## 6. Requisiti di Sistema

- Ambiente: **Google Colab** (Target GPU: **NVIDIA Tesla T4**).
- Dipendenze: **CUDA Toolkit**, **Python** 3.x (Librerie: Rasterio, Folium, Matplotlib).
