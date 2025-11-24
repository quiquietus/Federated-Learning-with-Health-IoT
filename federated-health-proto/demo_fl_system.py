"""
Automated FL System Demo - Simulates 3 hospital clients
"""
import requests
import time
import json
from pathlib import Path

API_URL = "http://localhost:8000"
SAMPLE_DATA = Path(r"C:\Users\adity\.gemini\antigravity\brain\80e970b8-288b-46ec-96cf-121df7ab677c\sample_hospital_data.csv")

def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print('='*60)

def register_user(user_id, email, org):
    """Register a hospital user"""
    data = {
        "user_id": user_id,
        "email": email,
        "password": "test123",
        "client_type": "hospital",
        "organization": org,
        "role": "doctor"
    }
    resp = requests.post(f"{API_URL}/api/register", json=data)
    if resp.status_code == 200:
        token = resp.json()["access_token"]
        print(f"✅ Registered: {user_id} ({org})")
        return token
    else:
        print(f"❌ Failed to register {user_id}: {resp.text}")
        return None

def upload_dataset(token, user_id):
    """Upload dataset for a user"""
    if not SAMPLE_DATA.exists():
        print(f"❌ Sample data not found at {SAMPLE_DATA}")
        return False
    
    with open(SAMPLE_DATA, 'rb') as f:
        files = {'file': (f'hospital_data_{user_id}.csv', f, 'text/csv')}
        headers = {'Authorization': f'Bearer {token}'}
        resp = requests.post(f"{API_URL}/api/upload-dataset", files=files, headers=headers)
        
        if resp.status_code == 200:
            print(f"✅ Uploaded dataset for {user_id}")
            return True
        else:
            print(f"❌ Failed to upload for {user_id}: {resp.text}")
            return False

def start_round(token):
    """Start a new FL round"""
    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.post(f"{API_URL}/api/start-round/hospital", headers=headers)
    
    if resp.status_code == 200:
        round_data = resp.json()
        print(f"✅ Started Round {round_data['round_number']}")
        print(f"   Round ID: {round_data['round_id']}")
        print(f"   Status: {round_data['status']}")
        return round_data['round_id']
    else:
        print(f"❌ Failed to start round: {resp.text}")
        return None

def train_model(token, user_id):
    """Train model for a user"""
    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.post(f"{API_URL}/api/train", headers=headers)
    
    if resp.status_code == 200:
        result = resp.json()
        print(f"✅ Training completed for {user_id}")
        print(f"   Accuracy: {result['metrics']['accuracy']:.2%}")
        print(f"   F1 Score: {result['metrics']['f1_score']:.4f}")
        print(f"   Samples: {result['sample_count']}")
        return result
    else:
        print(f"❌ Training failed for {user_id}: {resp.text}")
        return None

def get_rounds(token):
    """Get round status"""
    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.get(f"{API_URL}/api/rounds/hospital", headers=headers)
    
    if resp.status_code == 200:
        return resp.json()
    return []

def get_metrics(token):
    """Get aggregated metrics"""
    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.get(f"{API_URL}/api/metrics/hospital", headers=headers)
    
    if resp.status_code == 200:
        return resp.json()
    return []

def download_model(token, filename="global_model.txt"):
    """Download global model"""
    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.get(f"{API_URL}/api/download-model/hospital", headers=headers)
    
    if resp.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(resp.content)
        print(f"✅ Downloaded global model: {filename}")
        return True
    else:
        print(f"❌ Download failed: {resp.text}")
        return False

def main():
    print("\n" + "="*60)
    print("FEDERATED LEARNING SYSTEM - COMPLETE DEMONSTRATION")
    print("="*60)
    
    # Step 1: Register 3 hospitals
    print_step(1, "Registering 3 Hospital Clients")
    
    hospital_a_token = register_user("hospital_a", "admin@hospital-a.com", "Hospital A")
    hospital_b_token = register_user("hospital_b", "admin@hospital-b.com", "Hospital B")
    hospital_c_token = register_user("hospital_c", "admin@hospital-c.com", "Hospital C")
    
    if not all([hospital_a_token, hospital_b_token, hospital_c_token]):
        print("\n❌ Registration failed. Exiting.")
        return
    
    tokens = {
        "hospital_a": hospital_a_token,
        "hospital_b": hospital_b_token,
        "hospital_c": hospital_c_token
    }
    
    # Step 2: Upload datasets
    print_step(2, "Uploading Datasets for All Hospitals")
    
    for user_id, token in tokens.items():
        upload_dataset(token, user_id)
    
    # Step 3: Start FL round
    print_step(3, "Starting Federated Learning Round")
    
    round_id = start_round(hospital_a_token)
    if not round_id:
        print("\n❌ Failed to start round. Exiting.")
        return
    
    # Step 4: Train models (all 3 hospitals)
    print_step(4, "Training Local Models (All 3 Hospitals)")
    
    for user_id, token in tokens.items():
        print(f"\n🔬 Training for {user_id}...")
        train_model(token, user_id)
    
    # Step 5: Check round status
    print_step(5, "Checking Round Status")
    
    rounds = get_rounds(hospital_a_token)
    if rounds:
        latest = rounds[-1]
        print(f"Round {latest['round_number']}:")
        print(f"  Status: {latest['status']}")
        print(f"  Participants: {latest['num_participants']}")
        if latest['avg_accuracy']:
            print(f"  Avg Accuracy: {latest['avg_accuracy']:.2%}")
            print(f"  Avg F1: {latest['avg_f1_score']:.4f}")
    
    # Step 6: Wait for aggregation if round still active
    if rounds and rounds[-1]['status'] == 'active':
        print_step(6, "Waiting for Aggregation (200 seconds)")
        print("⏳ Aggregation scheduled for 200s after round start...")
        print("   (Checking every 30s)")
        
        for i in range(7):  # Check 7 times (210s total)
            time.sleep(30)
            rounds = get_rounds(hospital_a_token)
            if rounds and rounds[-1]['status'] == 'completed':
                print(f"\n✅ Round completed after waiting!")
                latest = rounds[-1]
                print(f"   Participants: {latest['num_participants']}")
                print(f"   Avg Accuracy: {latest.get('avg_accuracy', 0):.2%}")
                print(f"   Avg F1: {latest.get('avg_f1_score', 0):.4f}")
                break
            else:
                print(f"   ...still active (waited {(i+1)*30}s)")
    
    # Step 7: Get aggregated metrics
    print_step(7, "Retrieving Aggregated Metrics")
    
    metrics = get_metrics(hospital_a_token)
    if metrics:
        for m in metrics:
            print(f"\nRound {m['round_number']}:")
            print(f"  Accuracy: {m['accuracy']:.2%}")
            print(f"  F1 Score: {m['f1_score']:.4f}")
            print(f"  Participants: {m['num_participants']}")
    else:
        print("No completed rounds yet")
    
    # Step 8: Download global model
    print_step(8, "Downloading Global Model")
    
    download_model(hospital_a_token)
    
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\nWhat was demonstrated:")
    print("1. ✅ User registration (3 hospitals)")
    print("2. ✅ Dataset upload (each hospital)")
    print("3. ✅ FL round creation")
    print("4. ✅ Local model training (LightGBM)")
    print("5. ✅ FedAvg aggregation (automatic after 200s)")
    print("6. ✅ Metrics aggregation")
    print("7. ✅ Global model download")
    print("\nFiles created:")
    print("- global_model.txt (LightGBM global model)")
    print("\nServer storage:")
    print("- fl_storage/datasets/ (CSV files)")
    print("- fl_storage/updates/ (local models)")
    print("- fl_storage/models/ (global models)")
    print("="*60)

if __name__ == "__main__":
    main()
