import os
import requests

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url)
    response.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print("Download completed.")

if __name__ == "__main__":
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    download_file(red_url, "data/raw/winequality-red.csv")
    download_file(white_url, "data/raw/winequality-white.csv")
