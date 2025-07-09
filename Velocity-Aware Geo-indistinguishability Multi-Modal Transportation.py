# contains multiple constant variables that result in a less verbose coding
import privkit as pk
import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import geodesic
from privkit.utils import constants
import matplotlib.pyplot as plt
import seaborn as sns
from privkit.utils import (
    geo_utils,
    constants,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tqdm
import os
# Applying different PPMs to the location data in each scenario
def apply_ppm_to_scenario(ppm, location_data):
    # privacy parameter to be used in the appliance of the PPMs
    epsilon = 0.016
    
    if ppm == 'planar_laplace':
        planar_laplace = pk.PlanarLaplace(epsilon=epsilon)
        location_data = planar_laplace.execute(location_data)
        quality_loss = location_data.data.get(constants.QUALITY_LOSS)
        return location_data, quality_loss

    elif ppm == 'clustering':
        r = np.log(4) / epsilon
        clustering = pk.ClusteringGeoInd(r=r, epsilon=epsilon)
        location_data = clustering.execute(location_data)
        quality_loss = location_data.data.get(constants.QUALITY_LOSS)
        return location_data, quality_loss

    elif ppm == 'adaptive':
        ws = 2
        delta1 = 124.29
        delta2 = 428.56
        adaptive = pk.AdaptiveGeoInd(epsilon=epsilon, ws=ws, delta1=delta1, delta2=delta2)
        location_data = adaptive.execute(location_data)
        quality_loss = location_data.data.get(constants.QUALITY_LOSS)
        return location_data, quality_loss

    elif ppm == 'va_gi':
        m = 10
        va_gi = pk.VAGI(epsilon=epsilon, m=m)
        location_data, quality_loss = va_gi.execute(location_data)
        return location_data, quality_loss

    return None, None
# ---- Parameters ----
epsilon = 0.016
file = 'geolife'
ppm = 'planar_laplace'
scenario = 'high_vu_high_vr'
target_transport_mode = 'car'

# ---- Load Dataset ----
dataset = pk.datasets.GeolifeDataset()
dataset.load_dataset()
all_data = dataset.data.data
labeled_data = all_data[all_data['label'].notnull()].copy()

# ---- Filter for one transport mode ----
mode_data = labeled_data[labeled_data['label'] == target_transport_mode].reset_index(drop=True)
print(f"[INFO] Total data points for mode '{target_transport_mode}': {len(mode_data)}")

if len(mode_data) == 0:
    print("[ERROR] No data found for the given transport mode.")
else:
    # ---- Wrap into LocationData ----
    location_data_per_mode = pk.LocationData(id_name=target_transport_mode)
    location_data_per_mode.data = mode_data

    # ---- Clone and apply PPM ----
    location_data = pk.LocationData(f'{scenario}_{ppm}')
    location_data.data = location_data_per_mode.data.copy()
    
    # DEBUG: Check shape before
    print(f"[DEBUG] Original data shape: {location_data.data.shape}")

    # ---- Apply PPM ----
    location_data, quality_loss_values = apply_ppm_to_scenario(ppm, location_data)

    # ---- Check output ----
    if quality_loss_values is not None:            
        # Save
        filepath = os.path.join(constants.output_folder, file, ppm)
        os.makedirs(filepath, exist_ok=True)
        filename = f'ql_{scenario}_e{int(epsilon * 1000)}'
        np.save(os.path.join(filepath, filename), quality_loss_values)
        print(f"[SUCCESS] Quality loss saved to: {os.path.join(filepath, filename)}.npy")
    else:
        print("[ERROR] 'QUALITY_LOSS' not found in data after applying PPM.")
