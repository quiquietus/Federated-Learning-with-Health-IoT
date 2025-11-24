import pandas as pd
import numpy as np
import os
import zipfile
import io
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(r"C:\Users\adity\.gemini\antigravity\brain\80e970b8-288b-46ec-96cf-121df7ab677c")
BASE_DIR = Path(r"d:\Projects\Mini Project\Lab\Federated-Learning-with-Health-IoT\federated-health-proto")

zips = {
    "hospital": "archive (4).zip",
    "clinic": "archive (2).zip",
    "lab": "archive (3).zip",
    "iot": "archive (1).zip"
}

def generate_synthetic_hospital(n=50):
    return pd.DataFrame({
        'age': np.random.randint(40, 95, n),
        'anaemia': np.random.randint(0, 2, n),
        'creatinine_phosphokinase': np.random.randint(20, 800, n),
        'diabetes': np.random.randint(0, 2, n),
        'ejection_fraction': np.random.randint(15, 80, n),
        'high_blood_pressure': np.random.randint(0, 2, n),
        'platelets': np.random.randint(150000, 400000, n),
        'serum_creatinine': np.round(np.random.uniform(0.5, 5.0, n), 1),
        'serum_sodium': np.random.randint(110, 150, n),
        'sex': np.random.randint(0, 2, n),
        'smoking': np.random.randint(0, 2, n),
        'time': np.random.randint(4, 280, n),
        'DEATH_EVENT': np.random.randint(0, 2, n)
    })

def generate_synthetic_clinic(n=50):
    return pd.DataFrame({
        'bmi': np.round(np.random.uniform(18.5, 40.0, n), 1),
        'cholesterol': np.random.randint(150, 300, n),
        'glucose': np.random.randint(70, 180, n),
        'blood_pressure_sys': np.random.randint(90, 160, n),
        'blood_pressure_dia': np.random.randint(60, 100, n),
        'heart_rate': np.random.randint(60, 100, n),
        'exercise_hours': np.random.randint(0, 10, n),
        'sleep_hours': np.random.randint(4, 10, n),
        'stress_level': np.random.randint(1, 10, n),
        'age': np.random.randint(20, 80, n),
        'health_status': np.random.randint(0, 2, n)
    })

def generate_synthetic_lab(n=50):
    # Simulating image metadata/features
    return pd.DataFrame({
        'filename': [f'img_{i}.jpg' for i in range(n)],
        'label': np.random.choice(['RBC', 'WBC', 'Platelets'], n),
        'feature_1': np.random.rand(n),
        'feature_2': np.random.rand(n)
    })

def generate_synthetic_iot(n=50):
    return pd.DataFrame({
        'steps_daily': np.random.randint(1000, 20000, n),
        'calories_burned': np.random.randint(1200, 3500, n),
        'active_minutes': np.random.randint(0, 120, n),
        'sedentary_hours': np.random.randint(2, 16, n),
        'heart_rate_avg': np.random.randint(50, 90, n),
        'heart_rate_max': np.random.randint(100, 190, n),
        'sleep_quality': np.random.randint(1, 10, n),
        'water_intake': np.round(np.random.uniform(0.5, 4.0, n), 1),
        'screen_time': np.round(np.random.uniform(1.0, 12.0, n), 1),
        'activity_level': np.random.randint(0, 3, n)
    })

def process_client_type(client_type):
    zip_name = zips[client_type]
    zip_path = BASE_DIR / zip_name
    
    df = None
    
    # Try to extract real data
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if csv_files:
                    target = max(csv_files, key=lambda x: z.getinfo(x).file_size)
                    with z.open(target) as f:
                        df = pd.read_csv(f)
                        print(f"✅ Loaded real data for {client_type}")
        except Exception as e:
            print(f"⚠️ Failed to extract {client_type}: {e}")
            
    # Fallback to synthetic
    if df is None or df.empty:
        print(f"⚠️ Using synthetic data for {client_type}")
        if client_type == "hospital": df = generate_synthetic_hospital(150)
        elif client_type == "clinic": df = generate_synthetic_clinic(150)
        elif client_type == "lab": df = generate_synthetic_lab(150)
        elif client_type == "iot": df = generate_synthetic_iot(150)
        
    # Shuffle and Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    chunks = np.array_split(df, 3)
    
    for i, chunk in enumerate(chunks):
        out_name = f"{client_type}_data_batch_{i+1}.csv"
        out_path = OUTPUT_DIR / out_name
        chunk.to_csv(out_path, index=False)
        print(f"  -> Saved {out_path}")

def main():
    print("🚀 Starting Data Generation...")
    for c_type in zips.keys():
        process_client_type(c_type)
    print("✅ Done!")

if __name__ == "__main__":
    main()
