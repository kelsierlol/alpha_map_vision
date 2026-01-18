
import os
import tarfile
from urllib.request import urlretrieve
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract_mvtec_bottle(data_dir="data"):
    """
    Downloads and extracts the MVTec AD 'bottle' category.
    """
    dataset_name = "bottle"
    # The official FTP server for MVTec AD
    url = f"ftp://guest:GU.205d@ftp.mvtec.com/MVTec_AD/{dataset_name}.tar.xz"
    
    dataset_dir = os.path.join(data_dir, "mvtec_ad")
    os.makedirs(dataset_dir, exist_ok=True)

    archive_path = os.path.join(dataset_dir, f"{dataset_name}.tar.xz")
    
    if not os.path.exists(archive_path):
        logging.info(f"Downloading {dataset_name}.tar.xz from {url}...")
        try:
            urlretrieve(url, archive_path)
            logging.info("Download complete.")
        except Exception as e:
            logging.error(f"Failed to download the dataset: {e}")
            # Clean up corrupted download
            if os.path.exists(archive_path):
                os.remove(archive_path)
            return None
    else:
        logging.info("Archive already exists.")

    extracted_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(extracted_path):
        logging.info(f"Extracting {archive_path}...")
        try:
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(path=dataset_dir)
            logging.info("Extraction complete.")
        except tarfile.TarError as e:
            logging.error(f"Failed to extract the archive: {e}")
            return None
    else:
        logging.info(f"'{extracted_path}' already exists. Skipping extraction.")
    
    logging.info(f"MVTec 'bottle' dataset is ready at: {extracted_path}")
    return extracted_path

if __name__ == "__main__":
    download_and_extract_mvtec_bottle()
