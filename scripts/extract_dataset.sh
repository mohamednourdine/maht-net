#!/bin/bash
# MAHT-Net Dataset Extraction Script (Bash Version)
# Simple script to extract dataset.zip and nested archives

set -e  # Exit on any error

# Configuration
INPUT_FILE="${1:-data/raw/dataset.zip}"
OUTPUT_DIR="${2:-data/processed}"
LOG_FILE="logs/dataset_extraction.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$OUTPUT_DIR"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_color() {
    echo -e "${2}${1}${NC}"
}

# Check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_color "❌ Error: File $1 does not exist" $RED
        exit 1
    fi
}

# Extract ZIP files
extract_zip() {
    local zip_file="$1"
    local extract_to="$2"

    if command -v unzip &> /dev/null; then
        print_color "📦 Extracting $(basename "$zip_file")..." $BLUE
        unzip -q "$zip_file" -d "$extract_to"
        log "Extracted ZIP: $zip_file to $extract_to"
    else
        print_color "❌ Error: unzip command not found" $RED
        exit 1
    fi
}

# Extract RAR files
extract_rar() {
    local rar_file="$1"
    local extract_to="$2"

    if command -v unrar &> /dev/null; then
        print_color "📦 Extracting $(basename "$rar_file")..." $BLUE
        unrar x -y "$rar_file" "$extract_to/"
        log "Extracted RAR: $rar_file to $extract_to"
    elif command -v unar &> /dev/null; then
        print_color "📦 Extracting $(basename "$rar_file")..." $BLUE
        unar -q "$rar_file" -o "$extract_to"
        log "Extracted RAR: $rar_file to $extract_to"
    else
        print_color "⚠️  Warning: No RAR extractor found (unrar/unar). Skipping RAR files." $YELLOW
        log "Warning: RAR file skipped (no extractor): $rar_file"
    fi
}

# Find and extract nested archives
extract_nested() {
    local dir="$1"
    local level="${2:-0}"
    local max_level=3

    if [ "$level" -gt "$max_level" ]; then
        print_color "⚠️  Maximum nesting level ($max_level) reached" $YELLOW
        return
    fi

    # Find archive files
    local archives=$(find "$dir" -type f \( -name "*.zip" -o -name "*.rar" \) 2>/dev/null)

    if [ -z "$archives" ]; then
        return
    fi

    print_color "🔍 Found archive files at level $level" $BLUE

    while IFS= read -r archive; do
        if [ -f "$archive" ]; then
            local archive_dir=$(dirname "$archive")

            case "$archive" in
                *.zip)
                    extract_zip "$archive" "$archive_dir"
                    ;;
                *.rar)
                    extract_rar "$archive" "$archive_dir"
                    ;;
            esac

            # Remove original archive after extraction
            rm -f "$archive"

            # Recursively extract nested archives in the same directory
            extract_nested "$archive_dir" $((level + 1))
        fi
    done <<< "$archives"
}

# Preserve original folder structure (no reorganization)
preserve_structure() {
    local base_dir="$1"

    print_color "📁 Preserving original folder structure..." $BLUE

    # Only remove empty directories that might have been created during extraction
    find "$base_dir" -type d -empty -delete 2>/dev/null || true

    log "Original folder structure preserved in $base_dir"
}

# Print summary
print_summary() {
    local output_dir="$1"

    print_color "\n============================================" $GREEN
    print_color "🦷 MAHT-Net Dataset Extraction Summary" $GREEN
    print_color "============================================" $GREEN

    echo "📁 Input file: $INPUT_FILE"
    echo "📂 Output directory: $OUTPUT_DIR"

    # Count total files and directories
    local total_files=$(find "$output_dir" -type f | wc -l)
    local total_dirs=$(find "$output_dir" -type d | wc -l)

    echo "� Total files: $total_files"
    echo "📁 Total directories: $total_dirs"

    # Show directory structure (top level only)
    if [ -d "$output_dir" ]; then
        echo ""
        echo "📋 Directory structure:"
        ls -la "$output_dir" | grep "^d" | awk '{print "  📁 " $9}' | grep -v "^\.\.$" | grep -v "^\.$"
    fi

    print_color "✅ Extraction completed successfully!" $GREEN
    print_color "✅ Original folder structure preserved!" $GREEN
    print_color "============================================" $GREEN
}

# Main execution
main() {
    print_color "🚀 Starting MAHT-Net dataset extraction..." $GREEN
    log "Starting extraction: $INPUT_FILE -> $OUTPUT_DIR"

    # Check input file
    check_file "$INPUT_FILE"

    # Extract main archive
    case "$INPUT_FILE" in
        *.zip)
            extract_zip "$INPUT_FILE" "$OUTPUT_DIR"
            ;;
        *.rar)
            extract_rar "$INPUT_FILE" "$OUTPUT_DIR"
            ;;
        *)
            print_color "❌ Error: Unsupported file format. Only ZIP and RAR are supported." $RED
            exit 1
            ;;
    esac

    # Extract nested archives
    extract_nested "$OUTPUT_DIR"

    # Preserve original folder structure
    preserve_structure "$OUTPUT_DIR"

    # Print summary
    print_summary "$OUTPUT_DIR"

    log "Extraction completed successfully"
}

# Show usage
show_usage() {
    echo "Usage: $0 [INPUT_FILE] [OUTPUT_DIR]"
    echo ""
    echo "Arguments:"
    echo "  INPUT_FILE   Path to the dataset archive (default: data/raw/dataset.zip)"
    echo "  OUTPUT_DIR   Output directory (default: data/processed)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Extract default dataset"
    echo "  $0 data/raw/custom.zip               # Extract custom dataset"
    echo "  $0 data/raw/dataset.zip data/custom  # Extract to custom directory"
    echo ""
    echo "Requirements:"
    echo "  - unzip (for ZIP files)"
    echo "  - unrar or unar (for RAR files)"
}

# Handle help argument
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# Run main function
main
