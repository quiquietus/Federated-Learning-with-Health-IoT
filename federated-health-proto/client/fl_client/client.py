"""
Main client orchestrator for federated learning.
Handles download of global model, local training, and upload of updates.
"""
import requests
import gzip
import json
import time
from typing import Dict, Optional
from pathlib import Path


class FederatedClient:
    """Main client for federated learning participation"""
    
    def __init__(
        self,
        server_url: str,
        token: str,
        client_type: str,
        config: Optional[Dict] = None
    ):
        self.server_url = server_url.rstrip('/')
        self.token = token
        self.client_type = client_type
        self.config = config or {}
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def download_global_model(self) -> Optional[bytes]:
        """Download latest global model from server"""
        url = f"{self.server_url}/api/global-model/{self.client_type}/latest"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 404:
                print("No global model available yet")
                return None
            else:
                print(f"Error downloading model: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error connecting to server: {e}")
            return None
    
    def upload_model_update(
        self,
        round_id: int,
        model_data: bytes,
        sample_count: int,
        metrics: Dict[str, float],
        training_time: float,
        use_compression: bool = True
    ) -> bool:
        """Upload model update to server"""
        url = f"{self.server_url}/api/client-update"
        
        # Optionally compress
        if use_compression:
            model_data = gzip.compress(model_data)
        
        # Prepare form data
        files = {'model_update': ('model_update.bin', model_data)}
        data = {
            'round_id': round_id,
            'sample_count': sample_count,
            'client_accuracy': metrics.get('accuracy'),
            'client_f1_score': metrics.get('f1_score'),
            'client_precision': metrics.get('precision'),
            'client_recall': metrics.get('recall'),
            'training_time_seconds': training_time,
            'compression_used': use_compression
        }
        
        try:
            response = requests.post(url, headers=self.headers, files=files, data=data)
            if response.status_code == 200:
                print(f"✓ Update uploaded successfully")
                return True
            else:
                print(f"Error uploading update: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error uploading: {e}")
            return False
    
    def get_active_round(self) -> Optional[int]:
        """Get current active round ID"""
        url = f"{self.server_url}/api/rounds/{self.client_type}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                rounds = response.json()
                for r in rounds:
                    if r['status'] in ['active', 'pending']:
                        return r['round_id']
            return None
        except Exception as e:
            print(f"Error getting rounds: {e}")
            return None
