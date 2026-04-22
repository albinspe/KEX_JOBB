import numpy as np
import cv2 
from scipy import ndimage 
from PIL import Image
import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
import pandas as pd
from tqdm import tqdm

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

def get_azimuthal_integration_1d(image_np):
    """
    Kärnfunktionen som komprimerar 2D-spektrumet till en 1D-vektor (Radiell profil).
    Den integrerar värdena över koncentriska cirklar (radier).
    """

    y_center, x_center = np.indices((image_np.shape))
    y_center = y_center - (image_np.shape[0] / 2.0)
    x_center = x_center - (image_np.shape[1] / 2.0)


    radii = np.hypot(y_center, x_center)

    radii_bins = np.round(radii).astype(int)


    radial_sum = ndimage.sum(image_np, radii_bins, index=np.arange(0, radii_bins.max() + 1))
    

    radial_count = np.bincount(radii_bins.ravel())
    

    radial_count[radial_count == 0] = 1 
    

    radial_profile = radial_sum / radial_count
    
    return radial_profile


def extract_fourier_features(pil_image):
    """
    Huvudfunktion för att extrahera Fourierns 1D-profil från en bild.
    """

    img_np = np.array(pil_image)

    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
        

    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    img_np = cv2.resize(img_np, (448, 448))    


    img_ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb) 
    y_channel = img_ycbcr[: ,: , 0]


    f_transform = np.fft.fft2(y_channel)
    f_shift = np.fft.fftshift(f_transform)

    magnitude_spectrum = np.abs(f_shift)
    

    epsilon = 1e-8
    magnitude_spectrum = np.log(magnitude_spectrum + epsilon)


    feature_vector_1d = get_azimuthal_integration_1d(magnitude_spectrum)

    return feature_vector_1d


def main():
    ds = create_dataset()
    ds = ds.class_encode_column("label")

    total_images = len(ds)

    all_features = []
    all_labels = []

    checkpoint_size = 10000
    output_folder = "Fourier_checkpoints"

    os.makedirs(output_folder, exist_ok = True)
    print(f"Startar extraktion av {total_images} bilder...")
    

    counter = 0
    for i, example in enumerate(tqdm(ds)):
        try:
  
            feat = extract_fourier_features(example["image"])
            all_features.append(feat)
            all_labels.append(example["label"])
        except Exception as e:
            counter += 1 
            all_features.append(np.zeros(318))
            all_labels.append(example["label"])

        if (i + 1) % checkpoint_size == 0:
            df_tmp = pd.DataFrame(all_features)
            df_tmp['label'] = all_labels
            checkpoint_path = os.path.join(output_folder, f"checkpoint_{i+1}.parquet")
            df_tmp.to_parquet(checkpoint_path)
    print("Antal exceptions ", counter)        
            

    print("Skapar slutgiltig Master-fil...")
    columns = [f"freq_{i}" for i in range(1, 319)]
    master_df = pd.DataFrame(all_features, columns=columns)
    master_df['label'] = all_labels
    
    master_df.to_parquet("openfake_master_fourier.parquet")
    print(f"Klart! Master-dataset sparat som 'openfake_master_fourier.parquet'")

if __name__ == "__main__":
    main()
