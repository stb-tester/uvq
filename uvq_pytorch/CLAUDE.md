# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This directory contains the PyTorch implementation of Universal Video Quality (UVQ), Google's no-reference perceptual video quality assessment model. This is an alternative implementation to the TensorFlow version in the parent directory, compatible with PyTorch 1.12.

## Running Inference

```bash
# Install dependencies (PyTorch 1.12 compatible)
pip install -r requirements.txt

# Basic inference
python inference.py <video_file> <video_length_seconds>

# With additional options
python inference.py Gaming_1080P-0ce6_orig.mp4 20 --output results.txt --transpose
```

## Architecture

### Neural Network Components

- **CompressionNet** (`utils/compressionnet.py`): InceptionV1-based architecture analyzing compression artifacts
  - Input: 320x180x5 patches (5fps)
  - Model: `checkpoint/compressionnet_pytorch_statedict.pt`

- **ContentNet** (`utils/contentnet.py`): EfficientNet-based architecture for content analysis  
  - Input: 496x496x5 patches (5fps)
  - Model: `checkpoint/contentnet_pytorch.pt`
  - Labels: `checkpoint/contentnet_labels.csv` (3862 content categories)

- **DistortionNet** (`utils/distortionnet.py`): Distortion type analysis
  - Input: 640x360x1 patches (1fps)
  - Model: `checkpoint/distortionnet_pytorch_statedict.pt`

- **AggregationNet** (`utils/aggregationnet.py`): Combines features from multiple networks
  - Models: `checkpoint/aggregationnet_models/` (35 model variants)
  - Ensemble: 5 variants (ytugc20s_0 through ytugc20s_4) across 7 feature combinations

### Data Flow

1. Video → `video_reader.VideoReader.load_video()` → Two resized versions
2. Patches → Feature extraction networks → 16x16x100 feature maps
3. Features → AggregationNet ensemble → Quality scores for 7 combinations

## Key Implementation Details

### Model Loading
- CompressionNet and DistortionNet use `torch.load()` with state dictionaries
- ContentNet loads full model with `torch.load()`
- AggregationNet loads 35 models (5 ensembles × 7 feature combinations)

### Video Processing
- Uses shared `../utils/video_reader.py` for consistency with TensorFlow version
- Outputs two resolutions: standard (1280x720) and small for different networks
- Transpose operation supported via `--transpose` flag

### Feature Extraction
- All networks output 16x16x100 feature maps
- Features are combined by AggregationNet based on subnet combinations
- Final scores averaged across 5 ensemble models per combination

## Output Format

Returns dictionary with 7 quality scores:
- `compression`: CompressionNet only
- `content`: ContentNet only  
- `distortion`: DistortionNet only
- `compression_content`: Combined compression + content
- `compression_distortion`: Combined compression + distortion
- `content_distortion`: Combined content + distortion
- `compression_content_distortion`: All three networks (default/best score)