#!/bin/bash
# ============================================================
# COARSE SCAN - Passe 1
# Scan Qx de 20.05 VERS 20.00 par steps de 0.005
# Qy fixe (à adapter)
# ============================================================

QY=20.18
OUTPUT_DIR="outputs"
SCRIPT="run.py"

mkdir -p "$OUTPUT_DIR"

# Générer la grille avec Python (bash gère mal les floats)
QX_VALUES=($(python3 -c "
import numpy as np
vals = np.arange(20.05, 19.999, -0.005)
for v in vals:
    print(f'{v:.4f}')
"))

TOTAL=${#QX_VALUES[@]}
COUNT=0

echo "Coarse scan : $TOTAL points from Qx=20.05 to Qx=20.00 (step=0.005), Qy=$QY"
echo ""

for QX in "${QX_VALUES[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running Qx=$QX, Qy=$QY ..."

    OUTPUT_FILE="$OUTPUT_DIR/stopping_bandwidth_Qx${QX}_Qy${QY}.json"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "  -> Already exists, skipping."
        continue
    fi

    python "$SCRIPT" "$QX" "$QY"

    if [ -f "stopping_bandwidth_Qx${QX}_Qy${QY}.json" ]; then
        mv "stopping_bandwidth_Qx${QX}_Qy${QY}.json" "$OUTPUT_FILE"
    fi

    echo "  -> Done. Saved to $OUTPUT_FILE"
done

echo ""
echo "Coarse scan done. Results saved in $OUTPUT_DIR."