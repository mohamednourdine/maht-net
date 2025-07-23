#!/usr/bin/env python3
"""
MAHT-Net Dataset Extraction Script
Extracts dataset.zip and handles nested RAR/ZIP files

This script:
1. Extracts data/raw/dataset.zip
2. Finds and extracts any nested RAR or ZIP files
3. Organizes files into proper directory structure
4. Provides progress feedback and error handling

Usage:
    python scripts/extract_dataset.py
    python scripts/extract_dataset.py --input data/raw/custom.zip --output data/processed
"""

import os
import sys
import zipfile
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import logging

try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False
    print("Warning: rarfile not installed. RAR files will be skipped.")
    print("Install with: pip install rarfile")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetExtractor:
    """Handles extraction of nested ZIP and RAR archives"""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.extracted_files = []
        self.errors = []

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        Path('logs').mkdir(exist_ok=True)

    def is_archive(self, file_path: Path) -> bool:
        """Check if file is a supported archive format"""
        extensions = ['.zip', '.rar']
        return file_path.suffix.lower() in extensions

    def extract_zip(self, zip_path: Path, extract_to: Path) -> List[Path]:
        """Extract ZIP file and return list of extracted files"""
        extracted = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in the ZIP
                file_list = zip_ref.namelist()
                logger.info(f"Extracting {len(file_list)} files from {zip_path.name}")

                # Extract all files
                zip_ref.extractall(extract_to)

                # Return paths of extracted files
                for file_name in file_list:
                    extracted_file = extract_to / file_name
                    if extracted_file.exists():
                        extracted.append(extracted_file)

                logger.info(f"Successfully extracted {len(extracted)} files from {zip_path.name}")

        except zipfile.BadZipFile:
            error_msg = f"Error: {zip_path} is not a valid ZIP file"
            logger.error(error_msg)
            self.errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error extracting {zip_path}: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)

        return extracted

    def extract_rar(self, rar_path: Path, extract_to: Path) -> List[Path]:
        """Extract RAR file and return list of extracted files"""
        if not RARFILE_AVAILABLE:
            error_msg = f"Cannot extract {rar_path}: rarfile library not available"
            logger.warning(error_msg)
            self.errors.append(error_msg)
            return []

        extracted = []
        try:
            with rarfile.RarFile(rar_path, 'r') as rar_ref:
                # Get list of files in the RAR
                file_list = rar_ref.namelist()
                logger.info(f"Extracting {len(file_list)} files from {rar_path.name}")

                # Extract all files
                rar_ref.extractall(extract_to)

                # Return paths of extracted files
                for file_name in file_list:
                    extracted_file = extract_to / file_name
                    if extracted_file.exists():
                        extracted.append(extracted_file)

                logger.info(f"Successfully extracted {len(extracted)} files from {rar_path.name}")

        except rarfile.BadRarFile:
            error_msg = f"Error: {rar_path} is not a valid RAR file"
            logger.error(error_msg)
            self.errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error extracting {rar_path}: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)

        return extracted

    def extract_nested_archives(self, directory: Path, level: int = 0, max_level: int = 3):
        """Recursively extract nested archives"""
        if level > max_level:
            logger.warning(f"Maximum nesting level ({max_level}) reached. Skipping deeper extraction.")
            return

        # Find all archive files in the directory
        archive_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and self.is_archive(file_path):
                archive_files.append(file_path)

        if not archive_files:
            logger.info(f"No archive files found at level {level}")
            return

        logger.info(f"Found {len(archive_files)} archive files at level {level}")

        for archive_path in archive_files:
            # Extract directly into the parent directory
            extract_dir = archive_path.parent

            logger.info(f"Extracting {archive_path.name} to {extract_dir}")

            # Extract based on file type
            if archive_path.suffix.lower() == '.zip':
                extracted_files = self.extract_zip(archive_path, extract_dir)
            elif archive_path.suffix.lower() == '.rar':
                extracted_files = self.extract_rar(archive_path, extract_dir)
            else:
                continue

            self.extracted_files.extend(extracted_files)

            # Recursively extract any nested archives in the same directory
            if extracted_files:
                self.extract_nested_archives(extract_dir, level + 1, max_level)

            # Optionally remove the original archive after extraction
            try:
                archive_path.unlink()
                logger.info(f"Removed original archive: {archive_path.name}")
            except Exception as e:
                logger.warning(f"Could not remove {archive_path}: {str(e)}")

    def preserve_structure(self):
        """Preserve original folder structure (no reorganization)"""
        logger.info("Preserving original folder structure...")

        # Only remove empty directories that might have been created during extraction
        for root, dirs, files in os.walk(self.output_path, topdown=False):
            try:
                if not files and not dirs:
                    os.rmdir(root)
            except OSError:
                pass  # Directory not empty or other error

        logger.info(f"Original folder structure preserved in {self.output_path}")

    def extract(self):
        """Main extraction method"""
        if not self.input_path.exists():
            error_msg = f"Input file {self.input_path} does not exist"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False

        logger.info(f"Starting extraction of {self.input_path}")
        logger.info(f"Output directory: {self.output_path}")

        # Extract the main archive
        if self.input_path.suffix.lower() == '.zip':
            self.extracted_files = self.extract_zip(self.input_path, self.output_path)
        elif self.input_path.suffix.lower() == '.rar':
            self.extracted_files = self.extract_rar(self.input_path, self.output_path)
        else:
            error_msg = f"Unsupported archive format: {self.input_path.suffix}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False

        if not self.extracted_files:
            logger.error("No files were extracted from the main archive")
            return False

        # Extract nested archives
        self.extract_nested_archives(self.output_path)

        # Preserve original folder structure
        self.preserve_structure()

        return True

    def print_summary(self):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("🦷 MAHT-Net Dataset Extraction Summary")
        print("="*60)
        print(f"📁 Input file: {self.input_path}")
        print(f"📂 Output directory: {self.output_path}")
        print(f"📊 Total files extracted: {len(self.extracted_files)}")

        if self.output_path.exists():
            # Count total files and directories
            total_files = sum(1 for _ in self.output_path.rglob('*') if _.is_file())
            total_dirs = sum(1 for _ in self.output_path.rglob('*') if _.is_dir())

            print(f"📄 Total files: {total_files}")
            print(f"📁 Total directories: {total_dirs}")

            # Show directory structure (top level only)
            print("\n📋 Directory structure:")
            try:
                for item in sorted(self.output_path.iterdir()):
                    if item.is_dir():
                        print(f"  📁 {item.name}")
            except Exception as e:
                logger.warning(f"Could not list directory structure: {e}")

        print("✅ Extraction completed successfully!")
        print("✅ Original folder structure preserved!")
        print("="*60)

        if self.errors:
            print(f"⚠️  Errors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"   - {error}")
        else:
            print("✅ Extraction completed successfully!")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MAHT-Net dataset with nested archive support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract default dataset
  python scripts/extract_dataset.py

  # Extract custom dataset
  python scripts/extract_dataset.py --input data/raw/custom.zip --output data/processed

  # Extract with verbose logging
  python scripts/extract_dataset.py --verbose
        """
    )

    parser.add_argument(
        '--input', '-i',
        default='data/raw/dataset.zip',
        help='Input archive file path (default: data/raw/dataset.zip)'
    )

    parser.add_argument(
        '--output', '-o',
        default='data/processed',
        help='Output directory path (default: data/processed)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create extractor and run
    extractor = DatasetExtractor(args.input, args.output)

    try:
        success = extractor.extract()
        extractor.print_summary()

        if success and not extractor.errors:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
