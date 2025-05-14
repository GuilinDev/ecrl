from huggingface_hub import hf_hub_download
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_place_model(
    repo_id: str = "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
    filename: str = "onnx/model.onnx",
    output_dir: str = "../models",  # Relative to experiments/scripts
    model_subpath: str = "mobilenetv4/1"
):
    """
    Downloads an ONNX model from Hugging Face Hub and places it in the desired directory structure.

    Args:
        repo_id (str): The Hugging Face Hub repository ID.
        filename (str): The specific file to download from the repository (e.g., "onnx/model.onnx").
        output_dir (str): The base directory where the 'model_subpath' will be created.
                          This path is relative to the script's location if not absolute.
        model_subpath (str): The sub-directory structure (e.g., "mobilenetv4/1") where the model.onnx will be placed.
    """
    try:
        logging.info(f"Downloading {filename} from Hugging Face Hub repo: {repo_id}")

        # Construct the final directory path
        final_model_dir = os.path.join(output_dir, model_subpath)
        os.makedirs(final_model_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {os.path.abspath(final_model_dir)}")

        # Define the final path for the downloaded model
        local_model_file_path = os.path.join(final_model_dir, "model.onnx")

        # Download the model
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir_use_symlinks=False, # Avoid symlinks, copy the file directly
            local_dir=final_model_dir,    # Download directly into the target folder
            # Force rename to model.onnx if the downloaded filename is different within the subfolder
            # For 'onnx/model.onnx', filename in repo is 'model.onnx', so this isn't strictly needed
            # but good practice if filename in repo might differ.
        )
        
        # If hf_hub_download saved it with a different name based on its cache structure or if filename had subdirs,
        # ensure it's named model.onnx in the final_model_dir
        if os.path.basename(downloaded_path) != "model.onnx":
            logging.info(f"Renaming downloaded file from {downloaded_path} to {local_model_file_path}")
            os.rename(downloaded_path, local_model_file_path)
        else:
            # If it's already model.onnx and in the right place, downloaded_path should be local_model_file_path
             logging.info(f"Model downloaded to {downloaded_path}")

        if os.path.exists(local_model_file_path):
            logging.info(f"Model successfully downloaded and placed at: {local_model_file_path}")
        else:
            logging.error(f"Model download failed or file not found at expected path: {local_model_file_path}")

    except Exception as e:
        logging.error(f"An error occurred during model download: {e}", exc_info=True)

if __name__ == "__main__":
    # This script is intended to be run from experiments/scripts directory
    # The output_dir="../models" will correctly place the model in experiments/models/mobilenetv4/1/model.onnx
    download_and_place_model()
    logging.info("Ensure you have 'huggingface_hub' installed: pip install huggingface_hub") 