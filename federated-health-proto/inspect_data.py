import zipfile
import os
import pandas as pd
import numpy as np

BASE_DIR = r"d:\Projects\Mini Project\Lab\Federated-Learning-with-Health-IoT\federated-health-proto"
OUTPUT_DIR = r"C:\Users\adity\.gemini\antigravity\brain\80e970b8-288b-46ec-96cf-121df7ab677c"

zips = {
    "hospital": "archive (4).zip",
    "clinic": "archive (2).zip",
    "lab": "archive (3).zip",
    "iot": "archive (1).zip"
}

def inspect_zips():
    for key, filename in zips.items():
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                print(f"\n--- {key.upper()} ({filename}) ---")
                for name in zip_ref.namelist()[:5]: # Show first 5 files
                    print(name)
        else:
            print(f"❌ Missing {filename}")

if __name__ == "__main__":
    inspect_zips()
