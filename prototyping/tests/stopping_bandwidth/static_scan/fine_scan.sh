#!/bin/bash
# ============================================================
# FINE SCAN - Passe 2
# À lancer après avoir regardé les résultats du coarse scan
# et identifié la zone de variation.
#
# USAGE:
#   bash scan_fine.sh <Qx_min> <Qx_max> <step> <Qy>
#
# EXEMPLE:
#   bash scan_fine.sh 19.98 20.025 0.001 20.18
#
# NOTE: Qx=20.0 exact est automatiquement skippé (non matchable)
# ============================================================

if [ "$#" -ne 4 ]; then
    echo "Usage: bash scan_fine.sh <Qx_min> <Qx_max> <step> <Qy>"
    echo "Exemple: bash scan_fine.sh 19.98 20.025 0.001 20.18"
    exit 1
fi

QX_MIN=$1
QX_MAX=$2
STEP=$3
QY=$4
OUTPUT_DIR="outputs"
SCRIPT="run.py"

mkdir -p "$OUTPUT_DIR"

# Générer la grille avec Python
QX_VALUES=($(python3 -c "
import numpy as np
vals = np.arange(float('$QX_MIN'), float('$QX_MAX') + float('$STEP')/2, float('$STEP'))
for v in vals:
    print(f'{v:.4f}')
"))

TOTAL=${#QX_VALUES[@]}
COUNT=0

echo "Fine scan : $TOTAL points entre Qx=$QX_MIN et Qx=$QX_MAX (step=$STEP), Qy=$QY"
echo "Note: Qx=20.0000 sera skippé automatiquement."
echo ""

for QX in "${QX_VALUES[@]}"; do
    COUNT=$((COUNT + 1))

    # Skip Qx = 20.0 exact
    if [ "$QX" = "20.0000" ]; then
        echo "[$COUNT/$TOTAL] Qx=$QX -> SKIPPED (résonance entière exacte, non matchable)"
        continue
    fi

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
echo "Fine scan terminé !"