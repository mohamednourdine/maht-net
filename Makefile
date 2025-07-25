# MAHT-Net Makefile
# Multi-Stage Attention-enhanced Hybrid Transformer Network for Cephalometric Landmark Detection
#
# Konya Technical University
# PhD Research Project
# Developed by Mohamed Nourdine
# Supervisor: Dr. Öğr. Üyesi Betül Uzbaş

.PHONY: help setup install clean test train eval aws-setup deploy docs lint format

# Default target
help:
	@echo "           ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮"
	@echo "           │ 🦷  │ │ 🦷  │ │ 🦷  │ │ 🦷  │ │ 🦷  │ │ 🦷  │"
	@echo "           ╰─────╯ ╰─────╯ ╰─────╯ ╰─────╯ ╰─────╯ ╰─────╯"
	@echo ""
	@echo "              MAHT-Net: Multi-Stage Attention-enhanced"
	@echo "                 Hybrid Transformer Network for"
	@echo "               Cephalometric Landmark Detection"
	@echo ""
	@echo "🦷════════════════════════════════════════════════════════════════════════════🦷"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup          		- Complete project setup (recommended for first-time users)"
	@echo "  make install         		- Install Python dependencies"
	@echo "  make train          		- Start model training"
	@echo "  make eval           		- Evaluate trained model"
	@echo ""
	@echo "☁️  AWS Deployment:"
	@echo "  make aws-setup      		- Setup AWS EC2 environment"
	@echo "  make deploy         		- Deploy model to production"
	@echo ""
	@echo "📊 Data & Development:"
	@echo "  make extract-data   		- Extract dataset archives (bash script)"
	@echo "  make extract-data-py		- Extract dataset archives (Python script)"
	@echo "  make data-prep      		- Prepare datasets for training"
	@echo "  make test           		- Run all tests"
	@echo "  make lint           		- Run code quality checks"
	@echo "  make format         		- Format code with black and isort"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs           		- Generate documentation"
	@echo "  make clean          		- Clean temporary files and cache"
	@echo ""
	@echo "🦷════════════════════════════════════════════════════════════════════════════🦷"
	@echo "    🏛️  Konya Technical University | PhD Research by Mohamed Nourdine 🏛️"
	@echo "🦷════════════════════════════════════════════════════════════════════════════🦷"

setup:
	@echo "🔧 Setting up MAHT-Net project environment..."
	@echo "Accepting conda Terms of Service..."
	conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
	conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
	conda create -n maht-net python=3.9 -y
	@echo "✅ Conda environment 'maht-net' created"

	@echo "🔄 Updating package managers..."
	conda run -n maht-net pip install --upgrade pip setuptools wheel

# 	@echo "📦 Installing Python dependencies..."
# 	conda run -n maht-net pip install -r requirements.txt

	@echo "📁 Creating project directories..."
	mkdir -p data/{raw,processed,augmented,splits}
	mkdir -p models/{checkpoints,pretrained,configs}
	mkdir -p results/{experiments,ablation,clinical_validation}
	mkdir -p logs/{training,evaluation,deployment}
	mkdir -p scripts/{aws,deployment,evaluation}
	mkdir -p notebooks

	@echo ""
	@echo "🎉 MAHT-Net setup complete! ✅"
	@echo "📖 Next steps:"
	@echo "  1. Activate environment: conda activate maht-net"
	@echo "  2. Review documentation: documentation/00_executive_summary.md"
	@echo "  3. Setup AWS (if needed): make aws-setup"
	@echo "  4. Prepare data: make data-prep"


# Install dependencies
install:
	@echo "📦 Installing MAHT-Net dependencies..."
	conda run -n maht-net pip install --upgrade pip setuptools wheel
	conda run -n maht-net pip install -r requirements.txt
	@echo "Installation complete ✅"

# AWS EC2 Setup
aws-setup:
	@echo "☁️  Configuring AWS EC2 environment for MAHT-Net..."
	chmod +x scripts/aws/setup_ec2.sh
	./scripts/aws/setup_ec2.sh
	@echo "AWS EC2 setup complete ✅"

# Data extraction and preparation
extract-data:
	@echo "📦 Extracting dataset archives..."
	chmod +x scripts/extract_dataset.sh
	./scripts/extract_dataset.sh
	@echo "Dataset extraction complete ✅"

# Python-based extraction (with rarfile support)
extract-data-py:
	@echo "🐍 Extracting dataset with Python script..."
	python scripts/extract_dataset.py
	@echo "Python extraction complete ✅"

# Data preparation
data-prep:
	@echo "📋 Preparing cephalometric datasets..."
	python src/data/prepare_datasets.py --config configs/data_config.yaml
	python src/data/preprocess.py --augment --validate
	@echo "Data preparation complete ✅"

# Training
train:
	@echo "🚀 Starting MAHT-Net training..."
	python src/train.py --config configs/train_config.yaml --log-dir logs/training
	@echo "Training session logged to logs/training ✅"

# Evaluation
eval:
	@echo "Evaluating MAHT-Net performance..."
	python src/evaluate.py --config configs/eval_config.yaml --checkpoint models/checkpoints/best_model.pth
	@echo "Evaluation results saved to results/experiments ✅"

# Clinical validation
clinical-eval:
	@echo "🏥 Running clinical validation tests..."
	python src/clinical/validate.py --config configs/clinical_config.yaml
	@echo "Clinical validation complete ✅"

# Testing
test:
	@echo "🧪 Running MAHT-Net test suite..."
	python -m pytest tests/ -v --cov=src --cov-report=html
	@echo "Test results: htmlcov/index.html ✅"

# Code quality
lint:
	@echo "Running code quality checks..."
	flake8 src/ tests/
	pylint src/
	mypy src/
	@echo "Code quality check complete ✅"

# Code formatting
format:
	@echo "✨ Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "Code formatting complete ✅"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation: docs/_build/html/index.html ✅"

# Deploy to production
deploy:
	@echo "🚀 Deploying MAHT-Net to production..."
	chmod +x scripts/deployment/deploy.sh
	./scripts/deployment/deploy.sh
	@echo "Deployment complete ✅"

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf build/ dist/
	@echo "Cleanup complete ✅"

# Development environment
dev-setup: setup
	@echo "🛠️  Setting up development environment..."
	conda run -n maht-net pip install -r requirements-dev.txt
	conda run -n maht-net pre-commit install
	@echo "Development environment ready ✅"

# Monitor training
monitor:
	@echo "Starting TensorBoard monitoring..."
	tensorboard --logdir=logs/training --port=6006 --host=0.0.0.0
