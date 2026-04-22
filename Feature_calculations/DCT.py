import numpy as np
import cv2 
from scipy.fftpack import dct 
from skimage.util import view_as_blocks
from PIL import Image
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, load_from_disk
import os


def create_dataset():

    DATA_DIR = os.path.expanduser("~/data/openfake_arrow_chunks_50_50")

    chunk_dirs = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if d.startswith("chunk_")
    ])

    dataset_list = [load_from_disk(d) for d in chunk_dirs]
    full_dataset = concatenate_datasets(dataset_list)

    return full_dataset


def calculate_2d_dct(block):
    return dct(dct(block.T, norm = "ortho").T, norm = "ortho")


def extract_dct_features(PILIMAGE):
    img_np = np.array(PILIMAGE)

    img_ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    y_channel = img_ycbcr[ :, :, 0].astype(np.float32)

    y_centered = y_channel - 128.0


    h, w = y_centered.shape

    h_trunc = h - (h%8)
    w_trunc = w - (w&8)

    y_cropped = y_centered[: h_trunc, :w_trunc]

    blocks = view_as_blocks(y_cropped, block_shape = (8,8))

    num_blocks = blocks.shape[0] * blocks.shape[1]
    blocks_flat = blocks.reshape(num_blocks, 8, 8)
    
    dct_blocks = np.zeros_like(blocks_flat, dtype = np.float32)

    for i in range(num_blocks):
        dct_blocks[i] = calculate_2d_dct(blocks_flat[i])

    dct_coeffs = dct_blocks.reshape(num_blocks, 64)

    sigma_vals = np.std(dct_coeffs, axis = 0)

    beta_vals = sigma_vals / np.sqrt(2)
    beta_ac = beta_vals[1: ]
    return beta_ac


def main():
    ds = create_dataset()
    ds = ds.class_encode_column("label")

    total_images = len(ds)

    all_features = []
    all_labels = []

    checkpoint_size = 10000
    output_folder = "DCT_Checkpoints"

    os.makedirs(output_folder, exist_ok = True)
    print(f"Startar extraktion av {total_images} bilder...")
    
    for i, example in enumerate(tqdm(ds)):
        try:
            # Extrahera features (63 st Beta_AC)
            feat = extract_dct_features(example["image"])
            all_features.append(feat)
            all_labels.append(example["label"])
        except Exception as e:
            all_features.append(np.zeros(63))
            all_labels.append(example["label"])

        if (i + 1) % checkpoint_size == 0:
            df_tmp = pd.DataFrame(all_features)
            df_tmp['label'] = all_labels
            checkpoint_path = os.path.join(output_folder, f"checkpoint_{i+1}.parquet")
            df_tmp.to_parquet(checkpoint_path)
            

    print("Skapar slutgiltig Master-fil...")
    columns = [f"AC_{i}" for i in range(1, 64)]
    master_df = pd.DataFrame(all_features, columns=columns)
    master_df['label'] = all_labels
    
    master_df.to_parquet("openfake_master_dct.parquet")
    print(f"Klart! Master-dataset sparat som 'openfake_master_dct.parquet'")

if __name__ == "__main__":
    main()
