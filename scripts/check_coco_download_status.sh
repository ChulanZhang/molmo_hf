#!/bin/bash
# Script to check COCO Caption download status

source "$(dirname "$0")/../activate_env.sh" 2>/dev/null || true

MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-/anvil/scratch/x-pwang1/data/vlm/molmo}"
DOWNLOADS_DIR="${MOLMO_DATA_DIR}/torch_datasets/downloads"

echo "=========================================="
echo "COCO Caption Download Status Check"
echo "=========================================="
echo ""

# 1. Check if download process is running
echo "1. Download Process Status:"
if pgrep -f "download_data.py.*coco" > /dev/null; then
    echo "   ✓ Download process is RUNNING"
    ps aux | grep "download_data.py.*coco" | grep -v grep | awk '{print "     PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| Time:", $10}'
else
    echo "   ✗ Download process is NOT running"
fi
echo ""

# 2. Check downloaded files
echo "2. Downloaded Files in $DOWNLOADS_DIR:"
if [ -d "$DOWNLOADS_DIR" ]; then
    echo "   Files found:"
    ls -lh "$DOWNLOADS_DIR" 2>/dev/null | tail -20 | awk '{print "     " $9, "(" $5 ")"}'
    
    # Check specific files
    echo ""
    echo "   Specific file status:"
    for file in "annotations_trainval2014.zip" "train2014.zip" "val2014.zip"; do
        if [ -f "$DOWNLOADS_DIR/$file" ]; then
            SIZE=$(du -h "$DOWNLOADS_DIR/$file" | awk '{print $1}')
            echo "     ✓ $file exists ($SIZE)"
        else
            echo "     ✗ $file not found"
        fi
    done
else
    echo "   ✗ Downloads directory does not exist"
fi
echo ""

# 3. Check extracted files
echo "3. Extracted Files:"
if [ -d "$DOWNLOADS_DIR" ]; then
    # Check for extracted annotations
    if [ -d "$DOWNLOADS_DIR/annotations" ] || find "$DOWNLOADS_DIR" -name "captions_*.json" 2>/dev/null | head -1 | grep -q .; then
        echo "   ✓ Annotations extracted"
        find "$DOWNLOADS_DIR" -name "captions_*.json" 2>/dev/null | head -2 | while read f; do
            SIZE=$(du -h "$f" 2>/dev/null | awk '{print $1}')
            echo "     - $(basename $f) ($SIZE)"
        done
    else
        echo "   ✗ Annotations not extracted yet"
    fi
    
    # Check for extracted images
    if [ -d "$DOWNLOADS_DIR/train2014" ] || [ -d "$DOWNLOADS_DIR/val2014" ]; then
        echo "   ✓ Images extracted"
        if [ -d "$DOWNLOADS_DIR/train2014" ]; then
            COUNT=$(find "$DOWNLOADS_DIR/train2014" -name "*.jpg" 2>/dev/null | wc -l)
            echo "     - train2014: $COUNT images"
        fi
        if [ -d "$DOWNLOADS_DIR/val2014" ]; then
            COUNT=$(find "$DOWNLOADS_DIR/val2014" -name "*.jpg" 2>/dev/null | wc -l)
            echo "     - val2014: $COUNT images"
        fi
    else
        echo "   ✗ Images not extracted yet"
    fi
else
    echo "   ✗ Downloads directory does not exist"
fi
echo ""

# 4. Check disk space
echo "4. Disk Space:"
df -h "$DOWNLOADS_DIR" 2>/dev/null | tail -1 | awk '{print "   Available:", $4, "| Used:", $3, "| Total:", $2}'
echo ""

# 5. Check for active network/download activity
echo "5. Network Activity (if download is in progress):"
if pgrep -f "download_data.py.*coco" > /dev/null; then
    echo "   Checking for active network connections..."
    PID=$(pgrep -f "download_data.py.*coco" | head -1)
    if [ -n "$PID" ]; then
        # Check if process has open network connections
        if netstat -p 2>/dev/null | grep -q "$PID" || ss -p 2>/dev/null | grep -q "$PID"; then
            echo "   ✓ Process has active network connections (downloading)"
        else
            echo "   ⚠ Process running but no active network connections (may be extracting/processing)"
        fi
    fi
fi
echo ""

echo "=========================================="
echo "Note: COCO image files are very large (~13GB each)"
echo "Download may take a long time depending on network speed"
echo "=========================================="

