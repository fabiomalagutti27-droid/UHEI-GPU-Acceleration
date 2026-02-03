"""
PROJECT: Urban Heat Island Risk Map
FILE: scripts/generate_web_map.py
DESCRIPTION: Interactive geospatial visualization generator.
Creates web-based maps using Folium with risk overlay and administrative borders.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import subprocess

# Gestione import robusta per Colab
try:
    import folium
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "folium", "rasterio"])
    import folium
    import rasterio
    from rasterio.warp import transform_bounds

print("[INFO] Phase 12: Interactive geospatial visualization (Web GIS)")

# --- 1. CONFIGURAZIONE ---
BASE_PATH = "."
FILE_BIN_OUT = os.path.join(BASE_PATH, "output_risk_map.bin")
if not os.path.exists(FILE_BIN_OUT):
    FILE_BIN_OUT = os.path.join(BASE_PATH, "output_risk_map_32.bin")

FILE_REF_TIF = os.path.join(BASE_PATH, "ndvi_mean.tif")
FILE_BORDERS = os.path.join(BASE_PATH, "bologna_borders.geojson")
IMG_TEMP_PATH = os.path.join(BASE_PATH, "temp_heatmap_overlay.png")
HTML_OUT = os.path.join(BASE_PATH, "Mappa_Bologna_Completa.html")

WIDTH = 758
HEIGHT = 502

def create_interactive_map_with_borders():
    print(f"[INFO] Preparing interactive map (Dimensions: {WIDTH}x{HEIGHT})...")

    # 1. Caricamento Dati GPU
    if not os.path.exists(FILE_BIN_OUT):
        print(f"[ERROR] Missing output binary file ({FILE_BIN_OUT}).")
        return None
    
    try:
        raw_data = np.fromfile(FILE_BIN_OUT, dtype=np.float32)
        
        # Check dimensioni
        if raw_data.size != WIDTH * HEIGHT:
            print(f"[WARN] Dimension mismatch: found {raw_data.size}, expected {WIDTH*HEIGHT}. Attempting forced reshape.")
            if raw_data.size > WIDTH*HEIGHT:
                raw_data = raw_data[:WIDTH*HEIGHT]
            else:
                return None
            
        heatmap = raw_data.reshape((HEIGHT, WIDTH))
    except Exception as e:
        print(f"[ERROR] Data read error: {e}")
        return None
    
    # 2. Generazione Overlay PNG
    print("[INFO] Generating colored overlay...")
    valid_pixels = heatmap[heatmap > 0.001]
    
    if valid_pixels.size == 0:
        print("[WARN] Empty map. Unable to generate colors.")
        return None
    
    vmin = np.percentile(valid_pixels, 2)
    vmax = np.percentile(valid_pixels, 98)
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap('RdYlBu_r')
    
    colored_data = colormap(norm(heatmap))
    colored_data[heatmap < 0.001, 3] = 0.0  # Trasparenza
    
    plt.imsave(IMG_TEMP_PATH, colored_data)
    
    # 3. Coordinate GPS
    if not os.path.exists(FILE_REF_TIF):
        print(f"[ERROR] Missing reference file '{FILE_REF_TIF}'.")
        return None
    
    print("[INFO] Computing geographic coordinates...")
    with rasterio.open(FILE_REF_TIF) as src:
        left, bottom, right, top = transform_bounds(src.crs, {'init': 'epsg:4326'}, *src.bounds)
    
    center_lat = (bottom + top) / 2
    center_lon = (left + right) / 2
    
    # 4. Mappa Folium
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=False, control=True
    ).add_to(m)
    
    # 5. Overlay
    folium.raster_layers.ImageOverlay(
        image=IMG_TEMP_PATH,
        bounds=[[bottom, left], [top, right]],
        opacity=0.6,
        name="Rischio Termico (GPU)"
    ).add_to(m)
    
    # 6. Bordi
    if os.path.exists(FILE_BORDERS):
        print("[INFO] GeoJSON boundaries found.")
        try:
            with open(FILE_BORDERS, 'r') as f:
                geo_data = json.load(f)
            folium.GeoJson(
                geo_data, name="Confini",
                style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0}
            ).add_to(m)
        except:
            pass
    
    folium.LayerControl().add_to(m)
    
    # Salviamo su disco
    m.save(HTML_OUT)
    print(f"[INFO] Map saved to disk: {HTML_OUT}")
    
    return m

if __name__ == "__main__":
    create_interactive_map_with_borders()