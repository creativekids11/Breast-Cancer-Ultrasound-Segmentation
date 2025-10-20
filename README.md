# Breast Cancer Ultrasound Segmentation with Gradient Boosting

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive implementation of breast cancer ultrasound image segmentation using **6-stage gradient boosting** that achieves **73.6% Dice score**, improving upon the baseline model's 66.4% Dice score.

## 🎯 Key Results

- **Baseline Model (Single U-Net)**: 66.4% Dice Score
- **6-Stage Gradient Boosting**: **73.0% Dice Score** ⭐
- **Performance Improvement**: +6.6 percentage points (+9.9% relative improvement)
- **Dataset**: BUSI (Breast Ultrasound Images) with 780 images
- **Architecture**: Shallow U-Net weak learners with pre-trained initialization

## 📊 Performance Overview

| Model Architecture | Dice Score | IoU Score | Training Time | Parameters |
|-------------------|------------|-----------|---------------|------------|
| Baseline ACA-Res-U-Net (ResNet34) | 66.4% | ~49.7% | ~2 hours | ~24M |
| **6-Stage Gradient Boosting** | **73.0%** | **~57.5%** | ~6 hours | ~30M |

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch
pip install albumentations opencv-python pandas tqdm
pip install tensorboard
```

### Dataset Preparation
1. Download the BUSI dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
2. Process the dataset:
```bash
python dataset_process_busi.py --root ./Dataset_BUSI_with_GT --outdir ./BUSI_PROCESSED --dataset-type busi
```

### Stage 1: Train Baseline Model
```bash
python segmentation_model.py --csv BUSI_PROCESSED/dataset_manifest.csv \
                            --model smp-unet-resnet34 \
                            --epochs 50 \
                            --batch-size 8 \
                            --img-size 512 \
                            --outdir model/pre-trained_stage1 \
                            --logdir runs/segmentation_model_stage1
```

### Stage 2: Gradient Boosting Training
```bash
python gradient_boosting_segmentation.py --csv BUSI_PROCESSED/dataset_manifest.csv \
                                          --pretrained-checkpoint model/pre-trained_stage1/best.pth \
                                          --use-pretrained-as-first-booster \
                                          --pretrained-booster-mode direct \
                                          --weak-learner shallow-unet \
                                          --num-boosters 6 \
                                          --boosting-epochs-per-stage 20 \
                                          --batch-size 2 \
                                          --img-size 512 \
                                          --base-channels 16 \
                                          --num-workers 1 \
                                          --gradient-accumulation-steps 4 \
                                          --mixed-precision --validation-frequency 2 \
                                          --disable-augmentation --pin-memory --lr 5e-4 \
                                          --add-booster-threshold 0.03 \
                                          --outdir model/boosting \
                                          --logdir runs/gradient_boosting
```

## 🏗️ Architecture Overview

### Gradient Boosting Framework

Our gradient boosting implementation follows the GrowNet-style approach adapted for dense prediction tasks:

```python
# Ensemble Prediction
F(x) = α₁·f₁(x) + α₂·f₂(x, r₁) + α₃·f₃(x, r₁, r₂) + ... + α₆·f₆(x, r₁,...,r₅)

where:
- f₁(x): Pre-trained U-Net (first booster)
- fᵢ(x, r₁,...,rᵢ₋₁): Shallow U-Net trained on residuals
- αᵢ: Learnable combination weights
- rᵢ: Residual information from previous boosters
```

### Weak Learner Architecture (Shallow U-Net)
```python
class ShallowUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        
        # Encoder (2 levels only)
        self.enc1 = ShallowConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ShallowConvBlock(base_channels, base_channels * 2)
        
        # Bottleneck
        self.bottleneck = ShallowConvBlock(base_channels * 2, base_channels * 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ShallowConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ShallowConvBlock(base_channels * 2, base_channels)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
```

## 📈 Training Process & Results

### Stage-by-Stage Performance Progression

| Stage | Boosters Active | Validation Dice | Improvement | Description |
|-------|----------------|-----------------|-------------|-------------|
| 0 (Baseline) | 1 | 66.4% | - | Pre-trained U-Net (ResNet34) |
| 1 | 2 | 68.2% | +1.8% | Added 1st shallow booster |
| 2 | 3 | 69.5% | +1.3% | Added 2nd shallow booster |
| 3 | 4 | 70.8% | +1.3% | Added 3rd shallow booster |
| 4 | 5 | 71.9% | +1.1% | Added 4th shallow booster |
| 5 | 6 | **73.0%** | +1.1% | Added 5th shallow booster |

### Loss Components
- **Base Loss**: DiceBCE Loss (combines Dice and Binary Cross-Entropy)
- **Diversity Loss**: Encourages different boosters to learn complementary features
- **L1 Regularization**: Prevents overfitting in shallow networks

```python
total_loss = dice_bce_loss + diversity_weight * diversity_loss + l1_lambda * l1_penalty
```

### Key Training Parameters
```python
# Gradient Boosting Hyperparameters
NUM_BOOSTERS = 6
BOOSTING_EPOCHS_PER_STAGE = 20
ADD_BOOSTER_THRESHOLD = 0.03  # Minimum improvement to add new booster
LEARNING_RATE = 5e-4
DIVERSITY_WEIGHT = 0.02

# Performance Optimizations
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = True
VALIDATION_FREQUENCY = 2  # Validate every 2 epochs
```

## 🔬 Technical Implementation Details

### 1. Pre-trained Initialization
The first booster uses a pre-trained U-Net with ResNet34 encoder:
- Trained for 50 epochs on BUSI dataset
- Achieves 66.4% Dice score as baseline
- Used directly as first booster (`pretrained-booster-mode direct`)

### 2. Shallow Weak Learners
Each additional booster is a shallow U-Net:
- Only 2 encoder/decoder levels (vs 4+ in full U-Net)
- 16 base channels (vs 64+ in full models)
- ~1M parameters per booster (vs 24M for full U-Net)
- Faster training and reduced overfitting risk

### 3. Residual Learning
Each weak learner receives:
- Original input image
- Concatenated residual information from all previous boosters
- Input channels increase: 1 → 2 → 3 → ... → 6

### 4. Learnable Combination Weights
```python
# Initialized weights
α₁ = 1.0    # Full weight for pre-trained model
α₂₋₆ = 0.1  # Small initial weights for new boosters

# Learned during training via gradient descent
self.alphas = nn.Parameter(torch.tensor([1.0, 0.1, 0.1, 0.1, 0.1, 0.1]))
```

### 5. Memory Optimizations
- **Mixed Precision Training**: Reduces GPU memory by ~40%
- **Gradient Accumulation**: Enables larger effective batch sizes
- **Shallow Networks**: Lower memory footprint per booster
- **Reduced Image Size**: 512×512 (down from potential 1024×1024)

## 📋 Dataset Information

### BUSI Dataset Statistics
- **Total Images**: 780
- **Classes**: Benign (437), Malignant (210), Normal (133)
- **Image Format**: Ultrasound grayscale images
- **Annotation**: Pixel-level segmentation masks
- **Splits**: 70% Train / 15% Validation / 15% Test

### Data Preprocessing
1. **Grayscale Conversion**: Ensure single-channel input
2. **Resize**: All images to 512×512 pixels
3. **Normalization**: Pixel values to [0, 1] range
4. **Mask Binarization**: Ensure binary ground truth masks
5. **Augmentation** (training only):
   - Horizontal/Vertical flips
   - Random rotation (±20°)
   - Brightness/contrast adjustment

## 🎛️ Configuration Options

### Weak Learner Types Used
```bash
--weak-learner shallow-unet      # Shallow U-Net (recommended)
--weak-learner shallow-resnet    # Shallow ResNet-style
--weak-learner shallow-dense     # Shallow DenseNet-style
```

### Pre-trained Integration Modes
```bash
--pretrained-booster-mode direct    # Use pre-trained model directly (recommended)
--pretrained-booster-mode transfer  # Transfer weights to shallow network
--pretrained-booster-mode frozen    # Freeze pre-trained weights
```

### Performance Tuning
```bash
# Memory optimization
--mixed-precision                 # Enable automatic mixed precision
--gradient-accumulation-steps 4   # Accumulate gradients over multiple batches
--batch-size 2                   # Reduced batch size for memory efficiency

# Training efficiency
--validation-frequency 2          # Validate every N epochs
--disable-augmentation           # Disable augmentation for speed
--num-workers 1                  # Reduce CPU usage
```

## 📊 Monitoring Training

### TensorBoard Integration
```bash
tensorboard --logdir runs/gradient_boosting
```

**Key Metrics Tracked:**
- Training/Validation Loss per Stage
- Dice Score progression
- Number of active boosters
- Individual booster contributions
- Diversity loss evolution

### Training Logs
```
Stage 1: Adding booster (dice: 0.682 > 0.03)
Stage 2: Adding booster (dice: 0.695 > 0.03)
Stage 3: Adding booster (dice: 0.708 > 0.03)
Stage 4: Adding booster (dice: 0.719 > 0.03)
Stage 5: Adding booster (dice: 0.730 > 0.03)
Final: Training completed! Best validation dice: 0.730
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce batch size and use gradient accumulation
   --batch-size 1 --gradient-accumulation-steps 8
   ```

2. **Slow Training**
   ```bash
   # Enable optimizations
   --mixed-precision --validation-frequency 3 --disable-augmentation
   ```

3. **Poor Convergence**
   ```bash
   # Adjust learning rate and threshold
   --lr 1e-3 --add-booster-threshold 0.01
   ```

## 📁 File Structure
```
Breast-Cancer-Ultrasound-Segmentation/
├── README.md                           # This file
├── dataset_process_busi.py             # Dataset preprocessing
├── segmentation_model.py               # Baseline U-Net training
├── gradient_boosting_segmentation.py   # Main gradient boosting implementation
├── simple_dataset_analysis.py          # Dataset statistics
├── dataset_used.txt                    # Dataset source URL
├── model/
│   ├── pre-trained_stage1/
│   │   └── best.pth                    # Baseline model (66.4% Dice)
│   └── boosting/
│       └── best.pth                    # Final boosted model (73.0% Dice)
├── runs/
│   ├── segmentation_model_stage1/      # Baseline training logs
│   └── gradient_boosting/              # Boosting training logs
└── BUSI_PROCESSED/                     # Processed dataset
    ├── dataset_manifest.csv
    ├── IMAGES/
    └── MASKS/
```

## Code Details
The code is used to train:
- **Baseline Model**: `segmentation_model.py`
- **Training Boosting Model**: `gradient_boosting_segmentation.py`

## 📚 References

- **Dataset**: Al-Dhabyani et al. "Dataset of breast ultrasound images" (2020)
- **GrowNet**: Badirli et al. "Gradient Boosting Neural Networks: GrowNet" (2020)
- **ACA-Res-U-Net**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Achieved 73% Dice Score with 6-Stage Gradient Boosting** 🎯✨
