# Artifact Evaluation for "NetZIP: Algorithm/Hardware Co-design of In-network Lossless Compression for Distributed Large Model Training", MICRO 2025.

This section explains how to run experiments for the paper’s key results. We organize the steps into three parts: (1) collect and log intermediate values (gradients/activations) during fine-tuning to support measurements reported in Figure 5, and Figure 7; (2) evaluate compression ratios for LZ4/Snappy/Zstd/Deflate under NetZIP-algorithm variants (byte/bit grouping and delta-value compression), as shown in Figure 10; and (3) run the SimAI large-scale experiments to quantify end-to-end impact, corresponding to Figures 3, 4, and 13.

## 🔩 Requirements

### Hardware Requirements
- High-end CPU server with at least 1TB memory and 4TB storage space
- DGX node with H100 x 8 with at least 4TB storage space

### Software Requirements
- Python
- PyTorch
- Transformers
- Datasets
- NumPy
- LZ4, Snappy, Zlib, Zstandard
- Docker
- SimAI

## 📖 Contents
```
├── compression_ratio_analysis/     # NetZIP Compression ratio analysis
├── data_collection/                # Data collection from model fine-tuning
├── large_scale_simulation/         # Large-scale simulation scripts with SimAI
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ece-fast-lab/MICRO-2025-NetZIP.git
cd MICRO-2025-NetZIP
```

### 2. Setup Environment

#### Docker (Recommended)

```bash
# Run with NVIDIA PyTorch container
./start_docker.sh

# Inside container
cd /workspace/NetZIP
```

```bash
# Install dependencies and
pip install torch transformers datasets huggingface_hub numpy lz4 python-snappy zstandard tqdm
huggingface-cli login # enter the token and do not add token as git credential

```

## 📊 Usage Guide

### 1. Data Collection

Collect training data from large language models for analysis:

```bash
cd data_collection
./run1.sh
```

**Parameters:** (data_collection.py)
- `--model_name`: HuggingFace model identifier
- `--num_steps`: Number of training steps to collect

### 2. Compression Analysis

Analyze compression ratios for different algorithms:

```bash
cd compression_ratio_calculation
./run2.sh
```

**Parameters:** (compression_ratio_calculation.py)
- `--dir_step0`: Directory containing previous step data
- `--dir_step1`: Directory containing current step data
- `--output_dir`: Output directory for results
- `--num_workers`: Number of parallel workers

### 3. Large-Scale Simulation

Run large-scale simulations with different model sizes and network configurations:

```bash
cd large_scale_simulation
./get_SimAI.sh
cd script
./run3.sh
```

**Parameters:** (run.sh)
- **Model Size**: `7B`, `70B`, `175B`, or `405B` (model parameter count)
- **Topology**: `dcn_512` (data center network topology with 512 nodes)
- **Global Batch Size**: `128` (global batch size)
- **Bandwidths**: `100`, `200`, `400` Gbps (network bandwidth options)