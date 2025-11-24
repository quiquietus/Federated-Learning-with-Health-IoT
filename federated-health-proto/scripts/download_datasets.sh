#!/bin/bash
# Extract and organize pre-downloaded datasets

echo "=== Organizing Datasets ==="
echo "Extracting pre-downloaded dataset archives..."
echo ""

# Create data directory
mkdir -p data
cd data

echo "[1/4] Extracting Heart Failure dataset (Hospitals)..."
if [ -f "../archive (4).zip" ]; then
    unzip -o "../archive (4).zip" -d hospital_data
    echo "✓ Hospital data extracted"
else
    echo "⚠ archive (4).zip not found - skipping"
fi

echo "[2/4] Extracting BCCD Blood Cells dataset (Labs)..."
if [ -f "../archive (3).zip" ]; then
    unzip -o "../archive (3).zip" -d lab_data
    echo "✓ Lab data extracted"
else
    echo "⚠ archive (3).zip not found - skipping"
fi

echo "[3/4] Extracting Health Status dataset (Clinics)..."
if [ -f "../archive (2).zip" ]; then
    unzip -o "../archive (2).zip" -d clinic_data
    echo "✓ Clinic data extracted"
else
    echo "⚠ archive (2).zip not found - skipping"
fi

echo "[4/4] Extracting Student Lifestyle dataset (IoT)..."
if [ -f "../archive (1).zip" ]; then
    unzip -o "../archive (1).zip" -d iot_data
    echo "✓ IoT data extracted"
else
    echo "⚠ archive (1).zip not found - skipping"
fi

echo ""
echo "✓ Dataset extraction complete!"
echo "Datasets are organized in ./data/"
echo ""
echo "Structure:"
echo "  data/hospital_data/  - Heart failure clinical data"
echo "  data/lab_data/       - BCCD blood cell images"
echo "  data/clinic_data/    - Health status dataset"
echo "  data/iot_data/       - Student lifestyle time-series"

cd ..
