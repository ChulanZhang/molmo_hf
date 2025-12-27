#!/bin/bash
# Script to fix stuck COCO Caption download by killing process and cleaning up

echo "=========================================="
echo "COCO Caption Download Fix Script"
echo "=========================================="
echo ""

# Find stuck download process
PID=$(pgrep -f "download_data.py.*coco" | head -1)

if [ -n "$PID" ]; then
    echo "Found stuck process: PID $PID"
    echo "Killing process and its children..."
    pkill -P $PID 2>/dev/null
    kill $PID 2>/dev/null
    sleep 2
    
    # Force kill if still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing..."
        kill -9 $PID 2>/dev/null
    fi
    echo "✓ Process killed"
else
    echo "No stuck process found"
fi
echo ""

# Clean up lock files
echo "Cleaning up lock files..."
LOCK_FILE="/anvil/scratch/x-pwang1/data/vlm/huggingface/datasets/coco_caption/default/1.0.0_builder.lock"
if [ -f "$LOCK_FILE" ]; then
    rm -f "$LOCK_FILE"
    echo "✓ Removed builder lock file"
fi

DOWNLOAD_LOCK="/anvil/scratch/x-pwang1/data/vlm/huggingface/datasets/downloads/16054c8832f991725e3638c3bb1a49a8d97d0e2370926ff5d6e5a6c7c02f6d13.a36baaed21107d47d683e5b6f8b02a4afeb753a98746076896ada806272d520a.lock"
if [ -f "$DOWNLOAD_LOCK" ]; then
    rm -f "$DOWNLOAD_LOCK"
    echo "✓ Removed download lock file"
fi
echo ""

echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
echo ""
echo "You can now retry the download with:"
echo "  python scripts/download_data.py coco_caption --n_procs 1"
echo ""
echo "Note: Using --n_procs 1 (single process) is recommended to avoid"
echo "      multi-process deadlocks. The files are already downloaded,"
echo "      so it should just extract and process them quickly."
echo ""

