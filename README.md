# Multi-Frequency Memory Networks for Audio Event Detection (MF-Continuum-Conformer)

This repository contains a scalable, memory-augmented Conformer pipeline for sound event detection (SED) using multi-timescale neural memory modules.

## Overview
MF-Continuum-Conformer replaces static feed-forward networks with dynamic, test-time neural memory modules based on the Continuum Memory System (CMS). The architecture integrates Gamma (fast), Theta (medium), and Delta (slow) memory loops into a Conformer backbone for improved temporal modeling.

## Repository Structure
- `src/models/memory.py`: Momentum-based surprise updates and adaptive forgetting gates (Delta Rule)
- `src/models/mf_conformer.py`: Multi-frequency Conformer backbone with memory integration
- `src/models/convolution.py`: Standard Conformer convolution module
- `src/dataset.py`: Scalable dataloader (supports dummy and real data)
- `src/train.py`: Training loop for polyphonic event detection
- `src/evaluate.py`: PSDS evaluation stub for future benchmarking
- `configs/`: YAML configuration files
- `requirements.txt`: Python dependencies
- `run.sh`: Bash script to launch training

## Quick Start
```bash
pip install -r requirements.txt
bash run.sh  # or ./run.ps1 on Windows
```
