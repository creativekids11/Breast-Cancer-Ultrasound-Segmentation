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
                 input_channels=1, output_channels=1, pretrained_path=None, device='cuda'):
        super().__init__()
        
        self.num_boosters = num_boosters
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.device = device
        
        # Store weak learners
        self.weak_learners = nn.ModuleList()
        
        # Learnable combination weights (like GrowNet)
        self.alphas = nn.Parameter(torch.ones(num_boosters) * learning_rate)
        
        # Factory function to create weak learners
        self.weak_learner_factory = weak_learner_factory
        
        # Initialize first weak learner
        first_learner = weak_learner_factory(input_channels)
        
        # Load pretrained weights if provided
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            try:
                # Load checkpoint
                checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                
                # Smart loading: try to match layers by pattern and shape
                model_state = first_learner.state_dict()
                loaded_keys = []
                skipped_keys = []
                
                # First pass: exact matches
                for k, v in state_dict.items():
                    clean_key = k.replace('module.', '')
                    if clean_key in model_state and model_state[clean_key].shape == v.shape:
                        model_state[clean_key] = v
                        loaded_keys.append(clean_key)
                    else:
                        skipped_keys.append(k)
                
                # Second pass: try to match by layer patterns (e.g., first conv layers)
                if len(loaded_keys) == 0:  # If no exact matches, try pattern matching
                    print("No exact matches found, trying pattern-based loading...")
                    
                    # Get shallow network's conv layers (only weight parameters)
                    shallow_convs = [(k, model_state[k]) for k in model_state.keys() 
                                   if 'conv' in k and k.endswith('.weight') and len(model_state[k].shape) == 4]
                    
                    # Get pretrained conv layers (prioritize early layers)
                    pretrained_convs = [(k, v) for k, v in state_dict.items() 
                                      if 'conv' in k and k.endswith('.weight') and len(v.shape) == 4]
                    pretrained_convs.sort(key=lambda x: x[0])  # Sort by name
                    
                    print(f"Found {len(shallow_convs)} shallow conv layers, {len(pretrained_convs)} pretrained conv layers")
                    
                    # Try to match first few conv layers
                    for i, (shallow_key, shallow_weight) in enumerate(shallow_convs):
                        if i < len(pretrained_convs):
                            pretrained_key, pretrained_weight = pretrained_convs[i]
                            
                            # Check if shapes are compatible
                            shallow_shape = shallow_weight.shape
                            pretrained_shape = pretrained_weight.shape
                            
                            try:
                                if shallow_shape == pretrained_shape:
                                    model_state[shallow_key] = pretrained_weight
                                    loaded_keys.append(shallow_key)
                                    print(f"  ✓ Matched {shallow_key} <- {pretrained_key}")
                                elif (shallow_shape[1] == pretrained_shape[1] and 
                                      shallow_shape[0] <= pretrained_shape[0]):
                                    # Can use subset of output channels
                                    model_state[shallow_key] = pretrained_weight[:shallow_shape[0]]
                                    loaded_keys.append(shallow_key)
                                    print(f"  ✓ Adapted {shallow_key} <- {pretrained_key} ({pretrained_shape} -> {shallow_shape})")
                                elif (shallow_shape[1] == pretrained_shape[1] and 
                                      pretrained_shape[2] >= shallow_shape[2] and 
                                      pretrained_shape[3] >= shallow_shape[3]):
                                    # Try center crop for kernel size (e.g., 7x7 -> 3x3)
                                    h_diff = pretrained_shape[2] - shallow_shape[2]
                                    w_diff = pretrained_shape[3] - shallow_shape[3]
                                    h_start = h_diff // 2
                                    w_start = w_diff // 2
                                    cropped_weight = pretrained_weight[:, :, h_start:h_start+shallow_shape[2], w_start:w_start+shallow_shape[3]]
                                    
                                    # Handle output channel mismatch
                                    if cropped_weight.shape[0] > shallow_shape[0]:
                                        cropped_weight = cropped_weight[:shallow_shape[0]]
                                    elif cropped_weight.shape[0] < shallow_shape[0]:
                                        # Pad with zeros if needed (unlikely)
                                        pad_size = shallow_shape[0] - cropped_weight.shape[0]
                                        cropped_weight = torch.cat([cropped_weight, torch.zeros(pad_size, *cropped_weight.shape[1:], device=cropped_weight.device)], dim=0)
                                    
                                    model_state[shallow_key] = cropped_weight
                                    loaded_keys.append(shallow_key)
                                    print(f"  ✓ Cropped {shallow_key} <- {pretrained_key} ({pretrained_shape} -> {cropped_weight.shape})")
                                elif shallow_shape[1] == 1 and pretrained_shape[1] > 1:
                                    # Adapt single-channel to multi-channel by averaging
                                    adapted_weight = pretrained_weight.mean(dim=1, keepdim=True)
                                    if adapted_weight.shape[0] == shallow_shape[0]:
                                        model_state[shallow_key] = adapted_weight
                                        loaded_keys.append(shallow_key)
                                        print(f"  ✓ Adapted {shallow_key} <- {pretrained_key} (averaged channels)")
                            except Exception as e:
                                print(f"  ⚠ Failed to adapt {shallow_key}: {e}")
                                continue
                
                # Third pass: try to load batch norm layers if they exist
                for k, v in state_dict.items():
                    clean_key = k.replace('module.', '')
                    if ('bn' in clean_key or 'batch' in clean_key) and clean_key in model_state:
                        if model_state[clean_key].shape == v.shape:
                            model_state[clean_key] = v
                            loaded_keys.append(clean_key)
                
                first_learner.load_state_dict(model_state)
                
                total_keys = len(state_dict)
                print(f"✓ Loaded {len(loaded_keys)}/{total_keys} keys from pretrained model")
                
                if len(loaded_keys) == 0:
                    print("⚠ No compatible weights found. Using random initialization.")
                    print("   This is normal when loading complex models into simple architectures.")
                elif len(loaded_keys) < 5:
                    print("⚠ Only a few weights loaded. The pretrained model may have different architecture.")
                    print("   Continuing with partial initialization...")
                else:
                    print("✓ Successfully loaded pretrained weights!")
                    
            except Exception as e:
                print(f"⚠ Failed to load pretrained weights: {e}")
                print("Continuing with randomly initialized first learner")
        
        self.weak_learners.append(first_learner)
        
        # Track which boosters are active
        self.num_active_boosters = 1
        
    def add_weak_learner(self):
        """Add a new weak learner to the ensemble."""
        if self.num_active_boosters < self.num_boosters:
            # For segmentation, we can pass residuals as additional channels
            new_input_channels = self.input_channels + self.num_active_boosters
            new_learner = self.weak_learner_factory(new_input_channels)
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
        return current_pred

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
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model weights to initialize first booster")
    
    return parser.parse_args()

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
        input_channels=1,
        pretrained_path=args.pretrained,
        device=device
    ).to(device)
    
    print(f"Created gradient boosting model with {args.weak_learner} weak learners")
    
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
    
    print("Starting gradient boosting training...")
    trainer.run()

if __name__ == "__main__":
    main()
