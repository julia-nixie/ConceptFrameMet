# ConceptFrameMet

**Adaptive Source Domain Integration for Metaphor Detection**

This repository contains the implementation of `AdaptiveSourceQAMelBert`, a configurable framework for incorporating source domain information into metaphor detection models. The model extends MelBERT with flexible strategies for blending source domain predictions with target word embeddings.

## Overview

Metaphors often involve mapping concepts from a **source domain** (e.g., "war", "journey") to a **target domain**. This model uses a QA-based approach to predict source domains and integrates them adaptively into the metaphor detection pipeline.

### Key Features

- 🔄 **Configurable Blending Strategies**: Choose between replacement and additive modes
- 🎯 **Selective Application**: Apply source information only to metaphorical instances
- 📊 **Confidence Weighting**: Soft confidence scores for robust predictions
- 🔧 **Highly Flexible**: Multiple configuration flags for experimentation

## Architecture

```
INPUT SENTENCE → [Context Encoder] → target_context_embedding
                                             ↓
TARGET WORD → [QA Model] → Source Prediction + Confidence
                                             ↓
SOURCE WORD → [Isolated Encoder] → source_embedding
TARGET WORD → [Isolated Encoder] → target_embedding
                                             ↓
                        [ADAPTIVE BLENDING]
                   (replacement or additive mode)
                                             ↓
                        enhanced_embedding
                                             ↓
              [SPV + MIP] → [Classifier] → Metaphor Label
```

## Configuration Options

### 1. Source Blend Mode (`--source_blend_mode`)

Controls how source and target embeddings are combined:

- **`replacement`** (default): Original soft confidence approach
  ```
  blended = confidence × source + (1 - confidence) × target
  ```
  
- **`additive`**: Keeps target baseline, adds source as enhancement
  ```
  enhanced = target + α × confidence × source
  ```

### 2. Source Use Mode (`--source_use_mode`)

Controls when to apply source information:

- **`all`** (default): Use source for all samples
- **`metaphor_only`**: Only apply source to likely metaphors (based on preliminary score)

### 3. Source Alpha (`--source_alpha`)

Scaling factor for additive mode (default: `0.3`)
- Range: `[0.0, 1.0]`
- Lower values (0.1-0.3): Conservative
- Higher values (0.4-0.6): Aggressive

### 4. Metaphor Threshold (`--metaphor_threshold`)

Threshold for metaphor-only mode (default: `0.5`)
- Range: `[0.0, 1.0]`
- Controls when to apply source based on preliminary metaphor probability

### 5. Unfreeze Source QA (`--unfreeze_source_qa`)

Whether to fine-tune the source QA model (default: `False`)
- `False`: Freeze for efficiency
- `True`: Joint training (slower, more memory)

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7+
- Transformers 4.0+
- CUDA (recommended for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AdaptiveSourceQAMelBert.git
cd AdaptiveSourceQAMelBert

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training Example

```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode metaphor_only \
  --source_alpha 0.3 \
  --metaphor_threshold 0.5 \
  --do_train \
  --do_eval \
  --model_name roberta-base \
  --learning_rate 2e-5 \
  --num_train_epoch 10 \
  --per_gpu_train_batch_size 16
```

### Recommended Configurations

#### 1. **Safe Enhancement (Recommended Start)**
Best balance of safety and improvement:
```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode metaphor_only \
  --source_alpha 0.3 \
  --metaphor_threshold 0.5 \
  --do_train --do_eval
```

#### 2. **Conservative Approach**
Minimal risk, conservative source integration:
```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode all \
  --source_alpha 0.2 \
  --do_train --do_eval
```

#### 3. **Aggressive Enhancement**
Maximum source influence:
```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode additive \
  --source_use_mode metaphor_only \
  --source_alpha 0.5 \
  --metaphor_threshold 0.3 \
  --do_train --do_eval
```

#### 4. **Original Soft Confidence**
Baseline replacement mode:
```bash
python main.py \
  --model_type AdaptiveSourceQAMelBert \
  --source_blend_mode replacement \
  --source_use_mode all \
  --do_train --do_eval
```

For detailed usage examples and experiment recommendations, see [ADAPTIVE_SOURCE_QA_USAGE.md](ADAPTIVE_SOURCE_QA_USAGE.md).

## Expected Performance

Based on VUA18, VUA20, MOH-X, and TroFi benchmark datasets:

| Configuration | Use Case | Expected Gain |
|---------------|----------|---------------|
| Baseline MelBERT | Standard | - |
| Additive + All | Safe enhancement | +0.1-0.3% |
| **Additive + Metaphor-Only** | **Best overall** | **+0.2-0.6%** |
| Replacement + Metaphor-Only | Filtered soft | +0.1-0.4% |

See [ADAPTIVE_SOURCE_QA_USAGE.md](ADAPTIVE_SOURCE_QA_USAGE.md) for detailed performance targets.

## Model Components

### Core Files

- `modeling_qa_adaptive.py` - Main model implementation
- `main.py` - Training and evaluation script
- `data_loader.py` - Data loading utilities
- `modeling.py` - Base MelBERT model
- `run_classifier_dataset_utils.py` - Dataset utilities
- `main_config.cfg` - Configuration file

### Documentation

- `README.md` - This file
- `ADAPTIVE_SOURCE_QA_USAGE.md` - Detailed usage guide with examples

## Configuration File

You can also use a configuration file instead of command-line arguments:

```ini
[model]
model_type = AdaptiveSourceQAMelBert
source_blend_mode = additive
source_use_mode = metaphor_only
source_alpha = 0.3
metaphor_threshold = 0.5
unfreeze_source_qa = False

[training]
learning_rate = 2e-5
num_train_epoch = 10
per_gpu_train_batch_size = 16
early_stopping_patience = 5
```

Run with:
```bash
python main.py --config my_config.cfg --do_train --do_eval
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{adaptivesourceqamelbert2025,
  title={Adaptive Source Domain Integration for Metaphor Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/AdaptiveSourceQAMelBert}
}
```

## Related Work

This model extends:
- **MelBERT**: Choi et al., "MelBERT: Metaphor Detection via Contextualized Late Interaction using Metaphorical Identification Theories"
- Source domain prediction via QA-based approaches

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

This work builds upon the MetaphorFrame project and MelBERT architecture.
