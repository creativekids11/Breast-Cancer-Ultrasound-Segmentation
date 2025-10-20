#!/usr/bin/env python3
"""
gradient_boosting_segmentation.py

Implementation of Gradient Boosting with shallow networks (GrowNet-style) 
for breast cancer segmentation.
"""
from __future__ import annotations
import argparse
import os
import random
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision

# Import from your existing segmentation model
from segmentation_model import BreastSegDataset, dice_score, l1_regularization

# ----------------------------
# Shallow Network Architectures
# ----------------------------
class ShallowConvBlock(nn.Module):
    """Basic shallow convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class ShallowUNet(nn.Module):
    """Shallow U-Net with minimal depth for weak learners."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        
        # Encoder (only 2 levels)
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
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool1(e2))
        
        # Decoder
        d2 = self.up2(b)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Output
        out = self.out_conv(d1)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

class ShallowResBlock(nn.Module):
    """Shallow residual block for weak learners."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ShallowResNet(nn.Module):
    """Shallow ResNet-style network for weak learners."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_blocks=2):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Shallow residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ShallowResBlock(base_channels, base_channels))
        
        # Global average pooling + FC for lightweight processing
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels, base_channels)
        )
        
        # Upsample back to full resolution
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 1)
        )
        
    def forward(self, x):
        h, w = x.shape[2:]
        
        # Initial processing
        feat = self.initial(x)
        
        # Residual blocks
        for block in self.blocks:
            feat = block(feat)
        
        # Global context
        global_feat = self.gap(feat).view(feat.size(0), -1)
        global_feat = self.fc(global_feat).unsqueeze(-1).unsqueeze(-1)
        
        # Broadcast global features
        feat = feat + global_feat
        
        # Output
        out = self.upsample_conv(feat)
        return F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

class ShallowDenseNet(nn.Module):
    """Shallow DenseNet-style network."""
    def __init__(self, in_channels=1, out_channels=1, growth_rate=16, num_layers=4):
        super().__init__()
        
        self.initial = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        num_features = growth_rate
        
        for i in range(num_layers):
            self.dense_layers.append(nn.Sequential(
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, growth_rate, 3, padding=1)
            ))
            num_features += growth_rate
        
        # Output
        self.out_conv = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 1)
        )
        
    def forward(self, x):
        feat = self.initial(x)
        features = [feat]
        
        for layer in self.dense_layers:
            new_feat = layer(feat)
            features.append(new_feat)
            feat = torch.cat(features, dim=1)
        
        return self.out_conv(feat)

# ----------------------------
# Gradient Boosting Framework
# ----------------------------
class GradientBoostingSegmentation(nn.Module):
    """
    Gradient Boosting for segmentation using shallow networks.
    Similar to GrowNet but adapted for dense prediction tasks.
    """
    def __init__(self, weak_learner_factory, num_boosters=10, learning_rate=0.1, 
                 input_channels=1, output_channels=1):
        super().__init__()
        
        self.num_boosters = num_boosters
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        
        # Store weak learners
        self.weak_learners = nn.ModuleList()
        
        # Learnable combination weights (like GrowNet)
        # Initialize first booster with full weight (1.0), others with learning rate
        alphas_init = torch.ones(num_boosters) * learning_rate
        alphas_init[0] = 1.0  # Full weight for first (pre-trained) booster
        self.alphas = nn.Parameter(alphas_init)
        
        # Factory function to create weak learners
        self.weak_learner_factory = weak_learner_factory
        
        # Initialize first weak learner
        self.weak_learners.append(weak_learner_factory(input_channels))
        
        # Track which boosters are active
        self.num_active_boosters = 1
        
    def add_weak_learner(self):
        """Add a new weak learner to the ensemble."""
        if self.num_active_boosters < self.num_boosters:
            # For segmentation, we can pass residuals as additional channels
            new_input_channels = self.input_channels + self.num_active_boosters
            new_learner = self.weak_learner_factory(new_input_channels)
            
            # Ensure the new learner is on the same device as existing learners
            if len(self.weak_learners) > 0:
                device = next(self.weak_learners[0].parameters()).device
                new_learner = new_learner.to(device)
                print(f"  Moving new learner to device: {device}")
            
            self.weak_learners.append(new_learner)
            self.num_active_boosters += 1
            return True
        return False
        
    def forward(self, x, return_all_predictions=False):
        """Forward pass through all active weak learners."""
        h, w = x.shape[2:]
        batch_size = x.shape[0]
        
        predictions = []
        residuals = []
        
        # Initialize prediction
        current_pred = torch.zeros(batch_size, 1, h, w, device=x.device)
        current_input = x
        
        for i in range(self.num_active_boosters):
            # Get weak learner prediction
            weak_pred = self.weak_learners[i](current_input)
            
            # Apply learnable weight
            weighted_pred = self.alphas[i] * weak_pred
            predictions.append(weighted_pred)
            
            # Update ensemble prediction
            current_pred = current_pred + weighted_pred
            
            # Compute residual for next weak learner
            if i < self.num_active_boosters - 1:
                residual = weak_pred  # Use prediction as residual info
                residuals.append(residual)
                # Concatenate original input with residual information
                current_input = torch.cat([x] + residuals, dim=1)
        
        if return_all_predictions:
            return current_pred, predictions
        else:
            return current_pred
        
    def to(self, *args, **kwargs):
        """Override to() to ensure all weak learners are moved to the same device."""
        super().to(*args, **kwargs)
        # Ensure all weak learners are on the same device
        device = next(self.parameters()).device
        for i, learner in enumerate(self.weak_learners):
            self.weak_learners[i] = learner.to(device)
        return self

# ----------------------------
# Specialized Loss for Gradient Boosting
# ----------------------------
class GradientBoostingLoss(nn.Module):
    """
    Loss function for gradient boosting that encourages diversity
    and focuses on hard examples.
    """
    def __init__(self, base_loss_fn, diversity_weight=0.1, smooth=1e-5):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.diversity_weight = diversity_weight
        self.smooth = smooth
        
    def forward(self, predictions, targets, pos_weight=None, individual_preds=None):
        """
        Args:
            predictions: Final ensemble prediction
            targets: Ground truth masks
            pos_weight: Positive class weight
            individual_preds: List of individual weak learner predictions
        """
        # Base loss on ensemble prediction
        if hasattr(self.base_loss_fn, '__call__'):
            if pos_weight is not None:
                base_loss = self.base_loss_fn(predictions, targets, pos_weight)
            else:
                base_loss = self.base_loss_fn(predictions, targets)
        else:
            base_loss = F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=pos_weight)
        
        # Diversity loss to encourage different weak learners
        diversity_loss = torch.tensor(0.0, device=predictions.device)
        if individual_preds is not None and len(individual_preds) > 1:
            for i in range(len(individual_preds)):
                for j in range(i + 1, len(individual_preds)):
                    pred_i = torch.sigmoid(individual_preds[i])
                    pred_j = torch.sigmoid(individual_preds[j])
                    # Encourage diversity by penalizing similarity
                    similarity = F.cosine_similarity(pred_i.view(pred_i.size(0), -1), 
                                                   pred_j.view(pred_j.size(0), -1), dim=1)
                    diversity_loss += similarity.mean()
        
        total_loss = base_loss + self.diversity_weight * diversity_loss
        return total_loss

# ----------------------------
# Gradient Boosting Trainer
# ----------------------------
class GradientBoostingTrainer:
    """Trainer for gradient boosting segmentation."""
    
    def __init__(self, model, optimizer, scheduler, criterion, device, 
                 train_loader, val_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.writer = SummaryWriter(log_dir=args.logdir)
        self.best_val_dice = 0.0
        self.pos_weight = torch.tensor([args.pos_weight], device=device)
        
        # Boosting-specific parameters
        self.boosting_epochs_per_stage = args.boosting_epochs_per_stage
        self.add_booster_threshold = args.add_booster_threshold
        
        os.makedirs(args.outdir, exist_ok=True)
    
    def _train_epoch(self, epoch: int, stage: int):
        """Train one epoch for gradient boosting."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Stage {stage} Epoch {epoch}")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with all predictions
            ensemble_pred, individual_preds = self.model(imgs, return_all_predictions=True)
            
            # Compute loss
            loss = self.criterion(ensemble_pred, masks, pos_weight=self.pos_weight, 
                                individual_preds=individual_preds)
            
            # Add L1 regularization
            l1_penalty = l1_regularization(self.model, self.args.l1_lambda)
            total_loss_with_reg = loss + l1_penalty
            
            total_loss_with_reg.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", 
                           boosters=f"{self.model.num_active_boosters}")
        
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        self.writer.add_scalar("train/num_boosters", self.model.num_active_boosters, epoch)
        
        return avg_loss
    
    def _validate_epoch(self, epoch: int, stage: int):
        """Validate one epoch."""
        self.model.eval()
        val_loss, val_dice = 0, 0
        
        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, desc=f"Val Stage {stage}"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                
                ensemble_pred = self.model(imgs)
                loss = self.criterion(ensemble_pred, masks, pos_weight=self.pos_weight)
                val_loss += loss.item()
                
                preds_sigmoid = torch.sigmoid(ensemble_pred)
                val_dice += dice_score(preds_sigmoid, masks)
        
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_dice = val_dice / len(self.val_loader)
        
        self.writer.add_scalar("val/loss", avg_val_loss, epoch)
        self.writer.add_scalar("val/dice", avg_val_dice, epoch)
        
        # Save best model
        if avg_val_dice > self.best_val_dice:
            self.best_val_dice = avg_val_dice
            torch.save(self.model.state_dict(), os.path.join(self.args.outdir, "best.pth"))
        
        return avg_val_dice
    
    def run(self):
        """Main training loop with staged boosting."""
        stage = 0
        epoch = 0
        
        while self.model.num_active_boosters <= self.model.num_boosters:
            stage += 1
            print(f"\n{'='*50}")
            print(f"BOOSTING STAGE {stage}")
            print(f"Active boosters: {self.model.num_active_boosters}")
            print(f"{'='*50}")
            
            # Train current stage
            stage_best_dice = 0.0
            stage_start_epoch = epoch
            
            for stage_epoch in range(self.boosting_epochs_per_stage):
                epoch += 1
                
                # Train and validate
                train_loss = self._train_epoch(epoch, stage)
                val_dice = self._validate_epoch(epoch, stage)
                
                stage_best_dice = max(stage_best_dice, val_dice)
                
                print(f"Epoch {epoch}: Loss={train_loss:.4f}, "
                      f"Dice={val_dice:.4f}, Best={stage_best_dice:.4f}")
                
                if self.scheduler:
                    self.scheduler.step()
            
            # Decide whether to add another booster
            if (stage_best_dice > self.add_booster_threshold and 
                self.model.num_active_boosters < self.model.num_boosters):
                
                print(f"Adding booster (dice: {stage_best_dice:.4f} > {self.add_booster_threshold})")
                success = self.model.add_weak_learner()
                if success:
                    # Reset optimizer to include new parameters
                    self.optimizer.add_param_group({'params': self.model.weak_learners[-1].parameters()})
            else:
                print(f"Stopping boosting (dice: {stage_best_dice:.4f} <= {self.add_booster_threshold})")
                break
        
        self.writer.close()
        print(f"\nTraining completed! Best validation dice: {self.best_val_dice:.4f}")

# ----------------------------
# Main Functions
# ----------------------------
def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gradient Boosting Segmentation")
    
    # Data
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--img-size", type=int, default=512, help="Image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    
    # Model
    parser.add_argument("--weak-learner", choices=["shallow-unet", "shallow-resnet", "shallow-dense"], 
                        default="shallow-unet", help="Type of weak learner")
    parser.add_argument("--num-boosters", type=int, default=8, help="Maximum number of boosters")
    parser.add_argument("--learning-rate-boost", type=float, default=0.1, help="Boosting learning rate")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels for weak learners")
    
    # Pre-trained checkpoint
    parser.add_argument("--pretrained-checkpoint", type=str, default=None,
                        help="Path to pre-trained checkpoint (e.g., checkpoints/best.pth)")
    parser.add_argument("--use-pretrained-as-first-booster", action="store_true",
                        help="Use pre-trained model as first booster instead of random init")
    parser.add_argument("--pretrained-booster-mode", choices=["transfer", "direct", "frozen"], 
                        default="transfer",
                        help="How to use pre-trained model: transfer weights, use directly, or freeze weights")
    
    # Training
    parser.add_argument("--boosting-epochs-per-stage", type=int, default=15, 
                        help="Epochs to train each booster")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--add-booster-threshold", type=float, default=0.1, 
                        help="Minimum dice improvement to add booster")
    parser.add_argument("--pos-weight", type=float, default=11.0, help="Positive class weight")
    parser.add_argument("--l1-lambda", type=float, default=1e-5, help="L1 regularization")
    parser.add_argument("--diversity-weight", type=float, default=0.05, help="Diversity loss weight")
    
    # Output
    parser.add_argument("--outdir", default="checkpoints/gradient_boosting", help="Output directory")
    parser.add_argument("--logdir", default="runs/gradient_boosting", help="TensorBoard log directory")
    
    return parser.parse_args()

def transfer_weights_to_shallow_network(pretrained_model, shallow_model, method="adaptive"):
    """
    Transfer weights from pre-trained model to shallow network.

    Args:
        pretrained_model: Pre-trained model (e.g., UNet, ResNet)
        shallow_model: Target shallow network
        method: Transfer method - "adaptive", "direct", or "feature_extraction"

    Returns:
        shallow_model with transferred weights, or pretrained_model if transfer fails
    """
    print(f"Transferring weights using method: {method}")

    if method == "adaptive":
        # Adaptive weight transfer - map similar layers
        pretrained_state = pretrained_model.state_dict()
        shallow_state = shallow_model.state_dict()

        transferred_layers = 0
        total_layers = 0

        # Enhanced mapping for different architectures
        layer_mappings = {
            # UNet style mappings
            'encoder1': ['enc1', 'encoder1', 'down1', 'conv1'],
            'encoder2': ['enc2', 'encoder2', 'down2', 'conv2'],
            'decoder1': ['dec1', 'decoder1', 'up1', 'upconv1'],
            'decoder2': ['dec2', 'decoder2', 'up2', 'upconv2'],
            'bottleneck': ['bottleneck', 'bridge', 'center', 'middle'],

            # ResNet style mappings
            'conv1': ['initial', 'conv1', 'stem'],
            'layer1': ['blocks', 'layer1', 'resblock1'],
            'layer2': ['layer2', 'resblock2'],
            'layer3': ['layer3', 'resblock3'],
            'layer4': ['layer4', 'resblock4'],

            # Batch norm mappings
            'bn1': ['bn', 'batch_norm', 'norm1'],
            'bn2': ['bn2', 'norm2'],
            'bn3': ['bn3', 'norm3'],
        }

        for shallow_key in shallow_state.keys():
            total_layers += 1
            shallow_key_lower = shallow_key.lower()

            # Try different mapping strategies
            possible_keys = [shallow_key]  # Direct match first

            # Add mapped variations
            for pretrained_pattern, shallow_patterns in layer_mappings.items():
                if any(pattern in shallow_key_lower for pattern in shallow_patterns):
                    possible_keys.extend([k for k in pretrained_state.keys()
                                        if pretrained_pattern.lower() in k.lower()])

            # Try substring matching for common patterns
            for pretrained_key in pretrained_state.keys():
                pretrained_lower = pretrained_key.lower()
                if ('conv' in shallow_key_lower and 'conv' in pretrained_lower and
                    pretrained_state[pretrained_key].shape == shallow_state[shallow_key].shape):
                    possible_keys.append(pretrained_key)
                elif ('bn' in shallow_key_lower and 'bn' in pretrained_lower and
                      pretrained_state[pretrained_key].shape == shallow_state[shallow_key].shape):
                    possible_keys.append(pretrained_key)

            # Remove duplicates while preserving order
            seen = set()
            possible_keys = [x for x in possible_keys if not (x in seen or seen.add(x))]

            for possible_key in possible_keys[:10]:  # Limit to first 10 matches
                if possible_key in pretrained_state:
                    if pretrained_state[possible_key].shape == shallow_state[shallow_key].shape:
                        shallow_state[shallow_key] = pretrained_state[possible_key]
                        transferred_layers += 1
                        print(f"  ✓ {shallow_key} <- {possible_key}")
                        break
                    else:
                        print(f"  ⚠ Shape mismatch: {shallow_key} {shallow_state[shallow_key].shape} vs {possible_key} {pretrained_state[possible_key].shape}")

        print(f"Transferred {transferred_layers}/{total_layers} layers")

        # If very few layers transferred, fall back to feature extraction approach
        if transferred_layers / total_layers < 0.1:
            print("⚠ Very few layers transferred. Falling back to feature extraction approach.")
            return create_feature_extractor_booster(pretrained_model, shallow_model)

    elif method == "direct":
        # Direct transfer for exact matches
        pretrained_state = pretrained_model.state_dict()
        shallow_state = shallow_model.state_dict()

        transferred = 0
        for key in shallow_state:
            if key in pretrained_state and shallow_state[key].shape == pretrained_state[key].shape:
                shallow_state[key] = pretrained_state[key]
                transferred += 1

        print(f"Direct transfer: {transferred} layers matched")

    shallow_model.load_state_dict(shallow_state, strict=False)
    return shallow_model

def create_feature_extractor_booster(pretrained_model, shallow_model):
    """
    Create a booster that uses the pre-trained model for feature extraction
    and a shallow network for final prediction.
    """
    print("Creating feature extraction booster...")

    class FeatureExtractorBooster(nn.Module):
        def __init__(self, pretrained, shallow):
            super().__init__()
            self.pretrained = pretrained
            # Freeze pre-trained weights
            for param in self.pretrained.parameters():
                param.requires_grad = False

            self.shallow = shallow
            # Create adapter to match dimensions
            self.adapter = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            )

        def forward(self, x):
            # Use pre-trained model for feature extraction
            with torch.no_grad():
                features = self.pretrained(x)
            # Adapt features for shallow network
            adapted = self.adapter(features)
            # Final prediction with shallow network
            return self.shallow(adapted)

    return FeatureExtractorBooster(pretrained_model, shallow_model)

def load_pretrained_checkpoint(checkpoint_path, device):
    """
    Load pre-trained checkpoint from segmentation_model.py training.
    
    Returns:
        loaded_model: The pre-trained model
        model_info: Information about the loaded model
    """
    print(f"Loading pre-trained checkpoint: {checkpoint_path}")
    
    # Import the load function from segmentation_model
    from segmentation_model import load_model_from_checkpoint
    
    # Try to load with different model preferences
    preferred_models = ["smp-unet-resnet34", "aca-atrous-unet", "connect-unet", "aca-atrous-resunet"]
    
    loaded_model = None
    model_info = None
    
    for model_name in preferred_models:
        try:
            loaded_model, info, chosen = load_model_from_checkpoint(
                checkpoint_path, 
                preferred_model_name=model_name,
                device=device,
                img_size=512  # Default size
            )
            model_info = info
            print(f"Successfully loaded model as: {chosen}")
            break
        except Exception as e:
            print(f"Failed to load as {model_name}: {e}")
            continue
    
    if loaded_model is None:
        raise ValueError(f"Could not load checkpoint {checkpoint_path} with any known model type")
    
    return loaded_model, model_info

def create_weak_learner_factory(args):
    """Create factory function for weak learners."""
    
    def factory(input_channels):
        if args.weak_learner == "shallow-unet":
            return ShallowUNet(input_channels, 1, args.base_channels)
        elif args.weak_learner == "shallow-resnet":
            return ShallowResNet(input_channels, 1, args.base_channels, num_blocks=2)
        elif args.weak_learner == "shallow-dense":
            return ShallowDenseNet(input_channels, 1, growth_rate=args.base_channels//2, num_layers=3)
        else:
            raise ValueError(f"Unknown weak learner: {args.weak_learner}")
    
    return factory

def main():
    """Main function."""
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data loaders
    dataset = BreastSegDataset(args.csv, resize=(args.img_size, args.img_size), augment=True)
    val_len = int(len(dataset) * 0.2)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    # Create model
    weak_learner_factory = create_weak_learner_factory(args)
    model = GradientBoostingSegmentation(
        weak_learner_factory=weak_learner_factory,
        num_boosters=args.num_boosters,
        learning_rate=args.learning_rate_boost,
        input_channels=1
    ).to(device)
    
    # Load pre-trained checkpoint if specified
    if args.pretrained_checkpoint and args.use_pretrained_as_first_booster:
        print(f"\n{'='*60}")
        print("LOADING PRE-TRAINED CHECKPOINT")
        print(f"{'='*60}")
        
        try:
            # Load the pre-trained model
            pretrained_model, model_info = load_pretrained_checkpoint(args.pretrained_checkpoint, device)
            print(f"Pre-trained model info:\n{model_info}")
            
            if args.pretrained_booster_mode == "transfer":
                # Try weight transfer first
                first_booster = weak_learner_factory(input_channels=1)
                transferred_booster = transfer_weights_to_shallow_network(
                    pretrained_model, first_booster, method="adaptive"
                )
                
                # Check transfer success
                pretrained_state = pretrained_model.state_dict()
                transferred_state = transferred_booster.state_dict()
                transferred_count = sum(1 for k in transferred_state.keys() 
                                      if k in pretrained_state and 
                                      torch.equal(transferred_state[k], pretrained_state[k]))
                transfer_ratio = transferred_count / len(transferred_state)
                
                if transfer_ratio > 0.1:
                    model.weak_learners[0] = transferred_booster.to(device)
                    print("✓ Successfully transferred pre-trained weights to first booster!")
                else:
                    print("⚠ Weight transfer ineffective. Using pre-trained model directly.")
                    args.pretrained_booster_mode = "direct"
                    
            if args.pretrained_booster_mode == "direct":
                # Use pre-trained model directly as first booster
                class PretrainedBoosterWrapper(nn.Module):
                    def __init__(self, pretrained_model):
                        super().__init__()
                        self.model = pretrained_model
                        
                    def forward(self, x):
                        output = self.model(x)
                        # Handle case where model returns tuple (some models do this)
                        if isinstance(output, tuple):
                            return output[0]  # Return main output
                        return output
                
                model.weak_learners[0] = PretrainedBoosterWrapper(pretrained_model).to(device)
                print("✓ Using pre-trained model directly as first booster!")
                
            elif args.pretrained_booster_mode == "frozen":
                # Use pre-trained model with frozen weights
                class FrozenPretrainedBooster(nn.Module):
                    def __init__(self, pretrained_model):
                        super().__init__()
                        self.model = pretrained_model
                        # Freeze all weights
                        for param in self.model.parameters():
                            param.requires_grad = False
                            
                    def forward(self, x):
                        output = self.model(x)
                        # Handle case where model returns tuple (some models do this)
                        if isinstance(output, tuple):
                            return output[0]  # Return main output
                        return output
                
                model.weak_learners[0] = FrozenPretrainedBooster(pretrained_model).to(device)
                print("✓ Using pre-trained model as first booster (weights frozen)!")
            
            print(f"  Mode: {args.pretrained_booster_mode}")
            print("  Gradient boosting will start with your trained model and add shallow networks to correct errors.")
            
        except Exception as e:
            print(f"✗ Failed to load pre-trained checkpoint: {e}")
            print("  Continuing with randomly initialized boosters...")
    
    print(f"\nCreated gradient boosting model with {args.weak_learner} weak learners")
    if args.pretrained_checkpoint and args.use_pretrained_as_first_booster:
        print("  First booster initialized with pre-trained weights")
    
    # Loss and optimizer
    from segmentation_model import DiceBCELoss
    base_loss = DiceBCELoss(smooth=1e-5)
    criterion = GradientBoostingLoss(base_loss, diversity_weight=args.diversity_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.boosting_epochs_per_stage * args.num_boosters
    )
    
    # Trainer
    trainer = GradientBoostingTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )
    
    print("\nStarting gradient boosting training...")
    if args.pretrained_checkpoint and args.use_pretrained_as_first_booster:
        print("💡 Tip: Since you're using a pre-trained first booster, the initial dice score should be close to your checkpoint's performance!")
    
    trainer.run()

if __name__ == "__main__":
    main()
