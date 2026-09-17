#!/bin/bash
set -e

BUILD_DIR="${BUILD_DIR:-$PWD/librimix_build}"
SOURCES_DIR="$BUILD_DIR/sources"
OUTPUT_DIR="$BUILD_DIR/output"
DEST_DIR="${DEST_DIR:-$PWD/Libri2Mix_Official_Test}"

echo "============================================"
echo "  LibriMix Test Set Regeneration (Local Mac)"
echo "============================================"

mkdir -p "$SOURCES_DIR"

LS_TAR="$SOURCES_DIR/test-clean.tar.gz"
if [ ! -d "$SOURCES_DIR/LibriSpeech/test-clean" ]; then
    if [ ! -f "$LS_TAR" ]; then
        echo "[1/6] Downloading LibriSpeech test-clean (~350MB)..."
        curl -L -C - -o "$LS_TAR" https://www.openslr.org/resources/12/test-clean.tar.gz
    else
        echo "[1/6] LibriSpeech archive already exists, skipping download."
    fi
    echo "  Extracting..."
    tar -xzf "$LS_TAR" -C "$SOURCES_DIR"
else
    echo "[1/6] LibriSpeech test-clean already extracted, skipping."
fi

WHAM_ZIP="$SOURCES_DIR/wham_noise.zip"
if [ ! -d "$SOURCES_DIR/wham_noise/tt" ]; then
    if [ ! -f "$WHAM_ZIP" ]; then
        echo "[2/6] Downloading WHAM noise (~18GB, this will take a while)..."
        curl -L -C - -o "$WHAM_ZIP" https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
    else
        echo "[2/6] WHAM archive already exists, skipping download."
    fi
    echo "  Extracting only test (tt/) and metadata folders..."
    unzip -q -o "$WHAM_ZIP" "wham_noise/tt/*" "wham_noise/metadata/*" -d "$SOURCES_DIR"
else
    echo "[2/6] WHAM noise tt/ already extracted, skipping."
fi

LIBRIMIX_REPO="$BUILD_DIR/LibriMix"
if [ ! -d "$LIBRIMIX_REPO" ]; then
    echo "[3/6] Cloning LibriMix generator..."
    git clone https://github.com/JorisCos/LibriMix.git "$LIBRIMIX_REPO"
else
    echo "[3/6] LibriMix repo already cloned, skipping."
fi

echo "[4/6] Installing LibriMix requirements..."
pip3 install -q -r "$LIBRIMIX_REPO/requirements.txt"

LS_PATH="$SOURCES_DIR/LibriSpeech"
WHAM_PATH="$SOURCES_DIR/wham_noise"

if [ ! -d "$LS_PATH/test-clean" ]; then
    echo "ERROR: Could not find LibriSpeech test-clean at $LS_PATH/test-clean"
    exit 1
fi
if [ ! -d "$WHAM_PATH/tt" ]; then
    echo "ERROR: Could not find WHAM noise at $WHAM_PATH/tt"
    exit 1
fi

echo "  LibriSpeech: $LS_PATH"
echo "  WHAM noise:  $WHAM_PATH"

echo "[5/6] Generating LibriMix test set (this may take several minutes)..."
cd "$LIBRIMIX_REPO"

python3 scripts/create_librimix_from_metadata.py \
    --librispeech_dir "$LS_PATH" \
    --wham_dir "$WHAM_PATH" \
    --metadata_dir metadata/Libri2Mix \
    --librimix_outdir "$OUTPUT_DIR" \
    --n_src 2 \
    --freqs 16k \
    --modes min \
    --types mix_clean mix_both

GENERATED_AUDIO="$OUTPUT_DIR/Libri2Mix/wav16k/min/test"
GENERATED_META="$OUTPUT_DIR/Libri2Mix/metadata"

if [ ! -d "$GENERATED_AUDIO" ]; then
    echo "ERROR: Generation failed -- output directory not found: $GENERATED_AUDIO"
    exit 1
fi

echo "[6/6] Copying generated data to $DEST_DIR..."

for subdir in mix_clean mix_both s1 s2 noise; do
    src="$GENERATED_AUDIO/$subdir"
    dst="$DEST_DIR/sep_clean/$subdir"
    if [ -d "$src" ]; then
        echo "  Copying $subdir..."
        rm -rf "$dst"
        cp -r "$src" "$dst"
    fi
done

if [ -d "$GENERATED_META" ]; then
    echo "  Copying metadata..."
    mkdir -p "$DEST_DIR/metadata"
    cp "$LIBRIMIX_REPO/metadata/Libri2Mix/libri2mix_test-clean.csv" "$DEST_DIR/metadata/test.csv"
fi

MIX_COUNT=$(ls "$DEST_DIR/sep_clean/mix_clean/"*.wav 2>/dev/null | wc -l | tr -d ' ')
S1_COUNT=$(ls "$DEST_DIR/sep_clean/s1/"*.wav 2>/dev/null | wc -l | tr -d ' ')
S2_COUNT=$(ls "$DEST_DIR/sep_clean/s2/"*.wav 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "============================================"
echo "  VERIFICATION"
echo "============================================"
echo "  mix_clean: $MIX_COUNT files"
echo "  s1:        $S1_COUNT files"
echo "  s2:        $S2_COUNT files"

if [ "$MIX_COUNT" -ge 3000 ] && [ "$S1_COUNT" -ge 3000 ] && [ "$S2_COUNT" -ge 3000 ]; then
    echo "  SUCCESS: All 3000 files generated!"
else
    echo "  WARNING: Expected 3000 files in each folder."
fi
echo "============================================"

echo ""
echo "You can now delete the build directory to reclaim ~20GB:"
echo "  rm -rf $BUILD_DIR"
