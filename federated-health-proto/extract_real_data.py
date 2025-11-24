import zipfile
import os
import pandas as pd
import numpy as np
import io
from pathlib import Path

BASE_DIR = Path(r"d:\Projects\Mini Project\Lab\Federated-Learning-with-Health-IoT\federated-health-proto")
OUTPUT_DIR = Path(r"C:\Users\adity\.gemini\antigravity\brain\80e970b8-288b-46ec-96cf-121df7ab677c")

zips = {
    "hospital": "archive (4).zip",
    "clinic": "archive (2).zip",
    "lab": "archive (3).zip",
    "iot": "archive (1).zip"
}

def process_tabular(zip_path, client_type):
    print(f"Processing {client_type} from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the largest CSV
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            print(f"⚠️ No CSV found in {zip_path}")
            return
        
        target_csv = max(csv_files, key=lambda x: z.getinfo(x).file_size)
        print(f"  -> Found {target_csv}")
        
        with z.open(target_csv) as f:
            df = pd.read_csv(f)
            
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into 3
        chunks = np.array_split(df, 3)
        
        for i, chunk in enumerate(chunks):
            out_name = f"{client_type}_data_batch_{i+1}.csv"
            chunk.to_csv(OUTPUT_DIR / out_name, index=False)
            print(f"  -> Saved {out_name} ({len(chunk)} rows)")

def process_images(zip_path, client_type):
    print(f"Processing {client_type} from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Look for a CSV first (labels)
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        
        df = None
        if csv_files:
            target_csv = max(csv_files, key=lambda x: z.getinfo(x).file_size)
            print(f"  -> Found label CSV: {target_csv}")
            with z.open(target_csv) as f:
                df = pd.read_csv(f)
        else:
            print("  -> No CSV found, listing image files...")
            # List images and try to infer labels from folders
            images = [f for f in z.namelist() if f.endswith(('.jpg', '.jpeg', '.png'))]
            data = []
            for img in images:
                # Assuming folder structure like 'dataset/label/image.jpg'
                parts = img.split('/')
                if len(parts) > 1:
                    label = parts[-2]
                else:
                    label = 'unknown'
                data.append({'filename': img, 'label': label})
            df = pd.DataFrame(data)
            
        if df is not None and not df.empty:
             # Shuffle
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split into 3
            chunks = np.array_split(df, 3)
            
            for i, chunk in enumerate(chunks):
                out_name = f"{client_type}_data_batch_{i+1}.csv"
                chunk.to_csv(OUTPUT_DIR / out_name, index=False)
                print(f"  -> Saved {out_name} ({len(chunk)} rows)")
        else:
            print("⚠️ Could not extract data for Lab")

def main():
    with open("extraction.log", "w") as log:
        for client_type, filename in zips.items():
            path = BASE_DIR / filename
            log.write(f"Checking {path}...\n")
            if not path.exists():
                log.write(f"❌ Missing {filename}\n")
                continue
                
            try:
                if client_type == "lab":
                    process_images(path, client_type)
                else:
                    process_tabular(path, client_type)
                log.write(f"✅ Processed {client_type}\n")
            except Exception as e:
                log.write(f"❌ Error processing {client_type}: {e}\n")

if __name__ == "__main__":
    main()
