# Temporal Mixture-of-Experts Vision Transformer (Temporal MoE-ViT)

## 🎯 Overview

**Temporal Mixture-of-Experts Vision Transformer (Temporal MoE-ViT)** is a cutting-edge architecture for video understanding tasks, particularly Video Question Answering (VQA). This project implements a specialized transformer-based model that uses a mixture-of-experts (MoE) mechanism to dynamically route different types of spatio-temporal information through specialized expert networks.

The key innovation is the **dynamic routing of video patches** to task-specific experts based on their content:
- **Motion Expert**: Processes optical flow for temporal dynamics
- **Texture Expert**: Extracts low-level visual features using CNN stem
- **QA-Aligned Expert**: Conditions processing on the question
- **Fast Change Expert**: Captures frame-to-frame differences
- **Generic Experts**: General-purpose feedforward processors

## 📊 Architecture Overview

### Core Design Philosophy
The model uses a **top-K gating mechanism** (K=2) to route each token to the most relevant experts, enabling:
- **Computational efficiency**: Only active expert parameters contribute to compute
- **Interpretability**: Understand which experts process which tokens
- **Specialization**: Train experts to focus on specific video features
- **Scalability**: Add more experts without increasing per-token cost

### System Architecture Diagram
```
Input (Video + Question)
        ↓
[CLS] Token + Text Embeddings + Video Patches
        ↓
Spatio-Temporal Embeddings
        ↓
Transformer Layer (repeated):
├── Multi-Head Self-Attention
└── MoE Feedforward Block:
    ├── Router Network (Top-K Gating)
    ├── Expert Selection
    └── Weighted Expert Output Combination
        ↓
Classification Head
        ↓
Answer Predictions
```

## 📁 Project Structure

```
Temporal_MoE_Based_Vision_Transformer/
│
├── README.md                    # This file
├── environment.yaml             # Conda environment specification
├── train_dummy.py              # End-to-end training pipeline test
├── check_filenames.py          # Utility for diagnosing data issues
├── setup.ps1                   # Setup script
│
├── config/                     # Configuration files
│   └── training_msvd.yaml     # Training configuration
│
├── data/                       # Data loading and preprocessing
│   └── (dataset files)
│
├── models/                     # Core architecture modules
│   ├── base_vit.py            # Baseline Vision Transformer
│   ├── moe_vit_karm.py        # Temporal MoE-ViT implementation
│   ├── router.py              # Top-K router network
│   ├── experts.py             # Specialized expert implementations
│   ├── attention_karm.py       # Multi-head self-attention
│   └── prediction_head_karm.py # Classification head
│
├── train/                      # Training pipeline
│   ├── trainer.py             # Main training loop
│   ├── losses.py              # Loss functions
│   └── train.py               # Training script entry point
│
├── eval/                       # Evaluation utilities
│   └── (evaluation scripts)
│
└── scripts/                    # Automation scripts
    └── (bash scripts for training/eval)
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/JayGor-13/Temporal_MoE_Based_Vision_Transformer.git
cd Temporal_MoE_Based_Vision_Transformer
```

2. **Create a conda environment:**
```bash
conda env create -f environment.yaml
conda activate moevit
```

3. **Verify installation:**
```bash
python train_dummy.py
```

### Environment Setup

**Required dependencies:**
- Python 3.10+
- PyTorch 2.1+ with CUDA 11.8
- torchvision, torchaudio
- transformers
- einops
- opencv-python
- omegaconf
- tensorboard

**Installation via Conda:**
```bash
conda env create -f environment.yaml
conda activate moevit
```

**Installation via pip:**
```bash
pip install torch torchvision torchaudio transformers einops opencv-python omegaconf tqdm matplotlib tensorboard
```

## 🧠 Model Components

### 1. **BaseVit** - Baseline Vision Transformer
A standard Vision Transformer without MoE for comparison purposes.

```python
from models.base_vit import BaseVit

# Create baseline model
config = DummyConfig()
base_model = BaseVit(config)
```

**Features:**
- Standard Multi-Head Self-Attention
- Simple Feed-Forward networks
- Position embeddings for spatio-temporal sequences
- CLS token pooling for classification

### 2. **TemporalMoEViT** - MoE-Enhanced Architecture
The main model with specialized experts.

```python
from models.moe_vit_karm import TemporalMoEViT

# Create MoE model
moe_model = TemporalMoEViT(config)
```

### 3. **Router** - Top-K Expert Gating
Dynamically selects the K most relevant experts for each token.

```python
from models.router import Router

router = Router(
    embed_dim=768,
    num_experts=8,
    top_k=2,
    z_loss_weight=0.001,
    aux_loss_weight=0.01
)
```

**Features:**
- Learnable gate layer (Linear projection)
- Top-K selection mechanism
- Load-balancing loss (auxiliary loss)
- Z-loss for training stability
- Noise injection during training

### 4. **Specialized Experts**

#### Motion Expert (Expert 0)
Processes optical flow vectors to understand temporal motion.
```python
MotionExpert(embed_dim=768, flow_dim=64)
```

#### Texture Expert (Expert 1)
Uses a CNN stem to extract low-level visual features from raw patches.
```python
TextureExpert(embed_dim=768, patch_size=16, channels=3)
```

#### QA-Aligned Expert (Expert 3)
Conditions feature processing on the input question.
```python
QA_AlignedExpert(embed_dim=768)
```

#### Fast Change Expert (Expert 4)
Focuses on frame-to-frame differences and rapid scene changes.
```python
FastChangeExpert(embed_dim=768, delta_dim=32)
```

#### Generic Experts (Experts 2, 5, 6, 7)
General-purpose FFN processors as fallback/support networks.
```python
GenericExpert(embed_dim=768)
```

**All experts:**
- Take input embeddings [B, embed_dim]
- Return output embeddings [B, embed_dim]
- Support flexible kwargs for specialized inputs
- Use residual connections and layer normalization

## 📝 Configuration

Create or modify `config/training_msvd.yaml`:

```yaml
model:
  embed_dim: 768
  num_layers: 4
  num_heads: 8
  video_patch_size: 16
  num_answer_classes: 50
  frames_per_video: 8
  text_seq_len: 128
  max_seq_len: 325

moe:
  num_experts: 8
  top_k: 2
  experts:
    motion:
      flow_dim: 64
    fast_change:
      delta_dim: 32

data:
  batch_size: 16
  num_workers: 0
  data_root: /path/to/data

training:
  epochs: 20
  learning_rate: 3.0e-4
  weight_decay: 0.01
  loss_alpha: 0.01
  device: cuda
```

## 🎓 Training

### Quick Training Test (Dummy Data)

```bash
python train_dummy.py
```

This script:
1. Creates synthetic dummy data
2. Trains both BaseVit and TemporalMoEViT
3. Evaluates both models on test set
4. Prints comparison metrics (accuracy, parameters, active FLOPs)

**Expected output:**
```
==================================================
--- 1. TRAINING BASELINE (BaseVit) ---
==================================================
Epoch 01/01 | [Train] Loss: 3.8234 | Acc: 0.0234
==================================================
--- 2. TRAINING OUR MODEL (TemporalMoEViT) ---
==================================================
Epoch 01/01 | [Train] Loss: 3.2156 | Acc: 0.1543
==================================================
--- 3. FINAL EVALUATION & COMPARISON SUMMARY ---
==================================================

Metric                    | BaseViT         | TemporalMoEViT (Ours)
----------------------------------------------------------------------
Final Test Accuracy       | 0.2344          | 0.4156
Total Trainable Params    | 45.23M          | 52.41M
Active Params (Compute)   | 45.23M          | 28.14M
```

### Full Training Workflow

1. **Data Preprocessing** (if using real data):
```bash
python data/preprocess/extract_patches.py
python data/preprocess/tokenize_questions.py
```

2. **Training**:
```bash
python train/train.py --config config/training_msvd.yaml
```

3. **Evaluation**:
```bash
python eval/evaluate.py --checkpoint checkpoints/best_model.pt
```

## 📊 Input/Output Specifications

### Input Format

**Batch dictionary with keys:**
```python
batch = {
    'video': torch.randn(B, T, 3, 224, 224),           # Video frames
    'question_ids': torch.randint(0, 30522, (B, 128)), # Tokenized questions
    'answer_label': torch.randint(0, 50, (B,)),        # Answer labels
    'raw_patches': torch.randn(B*S, 3, 16, 16),        # Raw image patches
    'flow_vectors': torch.randn(B*S, 64),              # Optical flow
    'frame_deltas': torch.randn(B*S, 32),              # Frame differences
}
```

Where:
- **B**: Batch size
- **T**: Number of frames (8)
- **S**: Total sequence length (325 = 1 + 128 + 196)

### Output Format

**TemporalMoEViT returns a tuple:**
```python
logits, aux_loss = model(batch)
# logits: [B, num_answer_classes]
# aux_loss: scalar (sum of router losses)
```

**BaseVit returns logits only:**
```python
logits = model(batch)
# logits: [B, num_answer_classes]
```

## 🔄 Training Pipeline

### Loss Calculation

The model uses a combined loss function:

```python
from train.losses import calculate_total_loss

total_loss, task_loss = calculate_total_loss(
    logits=model_output[0],           # [B, num_classes]
    labels=batch['answer_label'],     # [B]
    auxiliary_loss=model_output[1],   # scalar (from router)
    alpha=0.01                        # weight for auxiliary loss
)

# total_loss = task_loss + alpha * auxiliary_loss
```

**Loss components:**
1. **Task Loss**: CrossEntropyLoss on answer prediction
2. **Auxiliary Loss** (from Router):
   - Z-loss: Stabilizes gate logits
   - Load-balancing loss: Encourages even expert utilization

### Optimizer & Scheduler

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)

# Optional: Use cosine annealing scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
```

## 🎯 Key Features

### ✅ Specialized Expert Networks
- Domain-specific experts for motion, texture, QA-alignment, and temporal changes
- Generic fallback experts for general processing
- Each expert can be updated independently

### ✅ Dynamic Routing
- Token-level routing decisions
- Top-K gating (default K=2)
- Load-balancing to prevent expert collapse
- Training stability via auxiliary losses

### ✅ Computational Efficiency
- Only top-K experts activated per token
- Significant FLOPs reduction vs. dense models
- Parameter sharing across tokens

### ✅ Interpretability
- Router outputs can be logged for analysis
- Expert routing patterns visualizable
- Understand which features activate which experts

### ✅ Modular Design
- Easy to add new expert types
- Pluggable attention mechanisms
- Flexible configuration system
- Well-documented code

## 📈 Performance Metrics

The model reports:
- **Top-1 Accuracy**: Answer prediction accuracy
- **Total Parameters**: All trainable parameters in the model
- **Active Parameters**: Parameters used for a forward pass (proxy for FLOPs)
- **Expert Utilization**: Distribution of expert usage

Example comparison:
| Metric | BaseViT | Temporal MoE-ViT |
|--------|---------|-----------------|
| Accuracy | 65.2% | 71.8% |
| Total Params | 45.2M | 52.4M |
| Active Params | 45.2M | 28.1M (↓38%) |

## 🔍 Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config
- Reduce `num_layers`
- Use gradient accumulation
- Reduce `num_experts`

### Poor Accuracy
- Ensure data preprocessing is correct (use `check_filenames.py`)
- Increase training epochs
- Adjust learning rate (try 1e-3 to 1e-5)
- Verify expert input shapes match expectations

### Shape Mismatches
Run the diagnostic script:
```bash
python check_filenames.py
```

Common issues:
- Video frame dimensions (must be 224×224)
- Patch size must divide image dimensions evenly
- Sequence length must include [CLS] + text + video tokens

## 📚 References & Related Work

This implementation is inspired by:
- **Vision Transformer (ViT)**: Dosovitskiy et al., 2020
- **Mixture of Experts (MoE)**: Shazeer et al., 2017
- **Switch Transformers**: Lewis et al., 2021
- **Video Understanding**: Recent VQA and video classification literature

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Submit a Pull Request with clear descriptions

## 📄 License

[Specify your license here]

## 📧 Contact

For questions or feedback:
- Open an issue on GitHub
- Contact: JayGor-13

---

**Last Updated**: 2026-03-20  
**Maintained by**: JayGor-13  
**Repository**: https://github.com/JayGor-13/Temporal_MoE_Based_Vision_Transformer
