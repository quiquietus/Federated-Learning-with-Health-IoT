#!/usr/bin/env python3
"""
Client registration script.
Registers a new user and stores the token for future use.
"""
import requests
import json
import sys
from pathlib import Path


def register_client(
    server_url: str,
    user_id: str,
    email: str,
    password: str,
    client_type: str,
    organization: str,
    role: str = "doctor"
):
    """Register a new client"""
    
    url = f"{server_url.rstrip('/')}/api/register"
    
    data = {
        "user_id": user_id,
        "email": email,
        "password": password,
        "client_type": client_type,
        "organization": organization,
        "role": role
    }
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 201:
            result = response.json()
            token = result['access_token']
            
            # Save token to file
            config_file = Path.home() / ".fl_client_config.json"
            config = {
                "server_url": server_url,
                "user_id": user_id,
                "client_type": client_type,
                "token": token
            }
            
            config_file.write_text(json.dumps(config, indent=2))
            
            print(f"✓ Registration successful!")
            print(f"User ID: {user_id}")
            print(f"Client Type: {client_type}")
            print(f"Token saved to: {config_file}")
            print(f"\nYour access token: {token}")
            
            return True
        else:
            print(f"✗ Registration failed: {response.status_code}")
            print(response.json())
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python register.py <server_url> <user_id> <email> <password> <client_type> <organization> [role]")
        print("Client types: hospital, clinic, lab, iot")
        print("Roles: doctor, insurance_analyst, other")
        sys.exit(1)
    
    server_url = sys.argv[1]
    user_id = sys.argv[2]
    email = sys.argv[3]
    password = sys.argv[4]
    client_type = sys.argv[5]
    organization = sys.argv[6]
    role = sys.argv[7] if len(sys.argv) > 7 else "doctor"
    
    register_client(server_url, user_id, email, password, client_type, organization, role)
