import zipfile
import os

try:
    with zipfile.ZipFile("archive (4).zip", 'r') as z:
        print("Success opening archive (4).zip")
        print(z.namelist()[:3])
except Exception as e:
    print(f"Error: {e}")
