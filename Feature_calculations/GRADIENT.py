import numpy as np
import cv2
from scipy.ndimage import convolve, rotate
from PIL import Image
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from datasets import load_from_disk, concatenate_datasets

# =====================================================================
# STEG 1: Generering av Multi-skala Gauss-masker (Formel 10 & 11)
# =====================================================================

def create_dataset():

    DATA_DIR = os.path.expanduser("~/data")

    chunk_dirs = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if d.startswith("chunk_")
    ])

    dataset_list = [load_from_disk(d) for d in chunk_dirs]
    full_dataset = concatenate_datasets(dataset_list)

    return full_dataset
    

def get_gaussian_compass_masks(sigma, size=5):
    """
    Genererar 8 st riktningsmasker baserat på derivatan av en förskjuten Gauss-funktion.
    Enligt Liu et al. (2023) Formel 10 och 11.
    """
    # Skapa ett koordinatnät
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # Formel 11: Standard Gaussian
    G = np.exp(-(xx**2 + yy**2) / (2. * sigma**2)) / (2 * np.pi * sigma**2)
    
    # Offset k (Enligt texten: "use 1/4 of the mask diameter for this offset")
    k = size / 4.0
    
    # Derivata av Gaussian i X-led med offset k
    xx_offset = xx + k
    G_prime = - (xx_offset) / (sigma**2) * np.exp(-(xx_offset**2 + yy**2) / (2. * sigma**2)) / (2 * np.pi * sigma**2)
    
    # Formel 10: Grundmasken M0
    M0 = G_prime * G
    
    # Rotera grundmasken 45 grader för att skapa de 8 kompassriktningarna
    masks = []
    for angle in range(0, 360, 45):
        M_rot = rotate(M0, angle, reshape=False, order=1)
        masks.append(M_rot)
        
    return masks

# =====================================================================
# STEG 2 & 3: GLDN och MGLH (Formel 12-18)
# =====================================================================
def extract_mglh_features(pil_image, target_size=(256, 256), sigmas=[1.0, 2.0], blocks_per_row=2):
    """
    Fullständig implementering av Multi-GLDN Histogram (MGLH).
    Extraherar features över flera skalor (sigmas) och block.
    """
    img_np = np.array(pil_image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # I paperet testade de PGGAN på 256x256 (Avsnitt 4.1.2)
    img_np = cv2.resize(img_np, target_size)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Fördela bilden i 3 separata kanaler: H, S, V
    channels = [img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]]
    
    all_histograms = []

    # Iterera över H, S och V
    for channel in channels:
        # Formel 8 & 9: Horisontell och vertikal gradient
        gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        
        gradients = [gx, gy]
        
        # Iterera över Gx och Gy
        for grad in gradients:
            # Iterera över de olika skalorna (Multi-scale n)
            for sigma in sigmas:
                # 1. Skapa maskerna för denna specifika skala
                masks = get_gaussian_compass_masks(sigma, size=5)
                
                # 2. Formel 15: Convolve
                responses = [convolve(grad, mask) for mask in masks]
                responses = np.stack(responses, axis=-1)
                
                # 3. Formel 13 & 14: Hitta max positiv och max negativ
                i_idx = np.argmax(responses, axis=-1)
                j_idx = np.argmin(responses, axis=-1)
                
                # 4. Formel 12: GLDN-koden
                gldn_code = 8 * i_idx + j_idx
                
                # 5. Formel 16 & 18: Block-uppdelning (Spatial aggregring)
                h, w = gldn_code.shape
                bh, bw = h // blocks_per_row, w // blocks_per_row
                
                for r in range(blocks_per_row):
                    for c in range(blocks_per_row):
                        block = gldn_code[r*bh : (r+1)*bh, c*bw : (c+1)*bw]
                        
                        # Beräkna 64-bins histogram för detta specifika block
                        hist, _ = np.histogram(block.flatten(), bins=64, range=(0, 63), density=True)
                        all_histograms.append(hist)

    # Limma ihop alla små histogram till en enda gigantisk vektor
    mglh_vector = np.concatenate(all_histograms)
    return mglh_vector

def process_single_image(example):
    try:
        # Inställningar enligt standard (2 sigmas, 2x2 blocks) ger 3072 features
        features = extract_mglh_features(example['image'], sigmas=[1.0, 2.0], blocks_per_row=2)
        return features
    except Exception as e:
        # Om bilden är korrupt, returnera nollor. 
        # (3 kanaler * 2 gradienter * 2 sigmas * 4 block * 64 bins = 3072 features)
        return [0.0] * 3072

# =====================================================================
# Huvudloop med Joblib och Checkpoints
# =====================================================================
def main():
    ds = create_dataset() # Er funktion för att ladda datan
    if "label" in ds.features and ds.features["label"].dtype != "int64":
         ds = ds.class_encode_column("label")

    total_images = len(ds)
    checkpoint_size = 10000
    output_folder = "gradient_checkpoints"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Totalt antal bilder i datasetet: {total_images}")
    
    # 3(HSV) * 2(Gx,Gy) * 2(sigmas) * 4(block) * 64(bins) = 3072
    feature_cols = [f"MGLH_{i}" for i in range(1, 3073)]
    all_dataframes = []

    for start_idx in range(0, total_images, checkpoint_size):
        end_idx = min(start_idx + checkpoint_size, total_images)
        print(f"\n--- Bearbetar bilder {start_idx} till {end_idx} ---")
        
        batch = ds.select(range(start_idx, end_idx))
        
        batch_features = Parallel(n_jobs=-1)(
            delayed(process_single_image)(example) for example in tqdm(batch, desc="Extraherar MGLH")
        )
        
        batch_labels = batch['label']
        
        df_batch = pd.DataFrame(batch_features, columns=feature_cols)
        df_batch['label'] = batch_labels
        
        checkpoint_path = os.path.join(output_folder, f"checkpoint_{end_idx}.parquet")
        df_batch.to_parquet(checkpoint_path)
        all_dataframes.append(df_batch)

    print("\nSlår ihop alla checkpoints till en slutgiltig Master-fil...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    master_df.to_parquet("openfake_master_gradient.parquet")
    print(f"KLART! ({len(master_df)} rader, 3072 features) sparat!")

if __name__ == "__main__":
    main()