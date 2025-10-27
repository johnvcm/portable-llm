import os
import requests
from huggingface_hub import hf_hub_download
import tqdm

# Path to save the model
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

try:
    print("Downloading DeepSeek Coder V2 Lite Instruct model (this may take a while)...")
    model_path = hf_hub_download(
        repo_id="CISCai/DeepSeek-Coder-V2-Lite-Instruct-SOTA-GGUF",
        filename="DeepSeek-Coder-V2-Lite-Instruct.IQ3_M.gguf", # IQ3_M é um bom equilíbrio entre tamanho e qualidade
        local_dir=models_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model successfully downloaded and saved to: {model_path}")
    print("\nEste é o modelo DeepSeek Coder V2 Lite Instruct com quantização IQ3_M.")
    print("Este modelo suporta contexto de até 128K tokens.")
except Exception as e:
    print(f"Error downloading the model: {e}")
    print("\nAlternative method: You can download the model manually from:")
    print("https://huggingface.co/CISCai/DeepSeek-Coder-V2-Lite-Instruct-SOTA-GGUF/blob/main/DeepSeek-Coder-V2-Lite-Instruct.IQ3_M.gguf")
    print(f"Then place it in the '{models_dir}' directory") 