# Dataset Organization Guide

## Pre-Downloaded Datasets

This project uses 4 pre-downloaded datasets. Place the following files in the main `federated-health-proto/` directory:

### Required Archive Files

1. **`archive (4).zip`** - Heart Failure Clinical Data (Hospitals)
   - Source: andrewmvd/heart-failure-clinical-data
   - Will be extracted to: `data/hospital_data/`

2. **`archive (3).zip`** - BCCD Blood Cell Images (Diagnostic Labs)
   - Source: orvile/bccd-blood-cell-count-and-detection-dataset
   - Will be extracted to: `data/lab_data/`

3. **`archive (2).zip`** - Health Status Dataset (Clinics)
   - Source: jacobhealth/health-status-dataset
   - Will be extracted to: `data/clinic_data/`

4. **`archive (1).zip`** - Student Lifestyle Dataset (IoT Device Hubs)
   - Source: steve1215rogg/student-lifestyle-dataset
   - Will be extracted to: `data/iot_data/`

## Extraction

Once all archive files are in place, run:

```bash
bash scripts/download_datasets.sh
```

This script will:
- Create a `data/` directory
- Extract each archive to its corresponding subdirectory
- Organize data for each client type
- Display the final directory structure

## Expected Structure After Extraction

```
federated-health-proto/
├── archive (1).zip
├── archive (2).zip
├── archive (3).zip
├── archive (4).zip
├── data/
│   ├── hospital_data/    # Extracted from archive (4).zip
│   ├── lab_data/         # Extracted from archive (3).zip
│   ├── clinic_data/      # Extracted from archive (2).zip
│   └── iot_data/         # Extracted from archive (1).zip
...
```

## Usage in Training

After extraction, you can reference datasets in training commands:

```bash
# Hospital client
python run_round.py --dataset ../data/hospital_data/[filename].csv

# Lab client  
python run_round.py --dataset ../data/lab_data/[images_dir]

# Clinic client
python run_round.py --dataset ../data/clinic_data/[filename].csv

# IoT client
python run_round.py --dataset ../data/iot_data/
```

## Notes

- The extraction script (`scripts/download_datasets.sh`) is safe to run multiple times
- If an archive file is missing, it will be skipped with a warning
- The `data/` directory is git-ignored
