#!/usr/bin/env python3
"""
train_with_boosting.py

Example training script demonstrating various boosting configurations
and how to use pretrained models.
"""
import os
import subprocess
import sys

def run_training(config_name, args):
    """Run training with given configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")
    
    cmd = ["python", "segmentation_model.py"] + args
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {config_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {config_name} failed with error: {e}")
        return False

def main():
    """Main function to run different training configurations."""
    
    # Base arguments
    base_args = [
        "--csv", "./BUSI_PROCESSED/dataset_manifest.csv",
        "--img-size", "512",
        "--batch-size", "8",  # Smaller batch size for boosting
        "--epochs", "50",     # Fewer epochs for demonstration
        "--lr", "1e-3",
        "--num-workers", "2"
    ]
    
    configs = []
    
    # 1. Baseline training (no boosting)
    configs.append((
        "Baseline (No Boosting)",
        base_args + [
            "--outdir", "checkpoints/baseline",
            "--logdir", "runs/baseline",
            "--model", "aca-atrous-resunet"
        ]
    ))
    
    # 2. Training with pretrained model (if available)
    if os.path.exists("checkpoints/best.pth"):
        configs.append((
            "With Pretrained Model",
            base_args + [
                "--outdir", "checkpoints/pretrained",
                "--logdir", "runs/pretrained", 
                "--model", "aca-atrous-resunet",
                "--pretrained", "checkpoints/best.pth"
            ]
        ))
    
    # 3. Ensemble Boosting
    configs.append((
        "Ensemble Boosting",
        base_args + [
            "--outdir", "checkpoints/ensemble",
            "--logdir", "runs/ensemble",
            "--model", "aca-atrous-resunet",
            "--enable-boosting",
            "--boost-type", "ensemble",
            "--num-boosters", "3"
        ]
    ))
    
    # 4. Cascade Boosting
    configs.append((
        "Cascade Boosting", 
        base_args + [
            "--outdir", "checkpoints/cascade",
            "--logdir", "runs/cascade",
            "--model", "aca-atrous-unet",  # Works better with cascade
            "--enable-boosting",
            "--boost-type", "cascade", 
            "--num-boosters", "3"
        ]
    ))
    
    # 5. Adaptive Boosting with Focal Loss
    configs.append((
        "Adaptive Boosting + Focal Loss",
        base_args + [
            "--outdir", "checkpoints/adaptive",
            "--logdir", "runs/adaptive",
            "--model", "aca-atrous-resunet",
            "--enable-boosting",
            "--boost-type", "adaptive",
            "--num-boosters", "3",
            "--adaptive-threshold", "0.3"
        ]
    ))
    
    # 6. Focal Loss Boosting
    configs.append((
        "Focal Loss Boosting",
        base_args + [
            "--outdir", "checkpoints/focal", 
            "--logdir", "runs/focal",
            "--model", "aca-atrous-resunet",
            "--enable-boosting",
            "--boost-type", "focal",
            "--num-boosters", "2"
        ]
    ))
    
    # 7. Ensemble with pretrained initialization (if available)
    if os.path.exists("checkpoints/best.pth"):
        configs.append((
            "Ensemble + Pretrained",
            base_args + [
                "--outdir", "checkpoints/ensemble_pretrained",
                "--logdir", "runs/ensemble_pretrained",
                "--model", "aca-atrous-resunet",
                "--enable-boosting",
                "--boost-type", "ensemble", 
                "--num-boosters", "3",
                "--pretrained", "checkpoints/best.pth"
            ]
        ))
    
    print("Available training configurations:")
    for i, (name, _) in enumerate(configs):
        print(f"{i+1}. {name}")
    
    print(f"{len(configs)+1}. Run all configurations")
    print(f"{len(configs)+2}. Quick test (check masks only)")
    
    try:
        choice = input(f"\nSelect configuration (1-{len(configs)+2}): ").strip()
        choice = int(choice)
        
        if choice == len(configs) + 2:
            # Quick test - just check masks
            print("Running quick mask check...")
            run_training("Mask Check", base_args + ["--check-masks", "--outdir", "temp_check"])
            return
        elif choice == len(configs) + 1:
            # Run all configurations
            print("Running all configurations...")
            for name, args in configs:
                success = run_training(name, args)
                if not success:
                    print(f"Stopping due to failure in {name}")
                    break
        elif 1 <= choice <= len(configs):
            # Run selected configuration
            name, args = configs[choice - 1]
            run_training(name, args)
        else:
            print("Invalid choice!")
            return
            
    except (ValueError, KeyboardInterrupt):
        print("Cancelled or invalid input.")
        return
    
    print("\nTraining completed! Check the results in:")
    print("- Checkpoints: checkpoints/*/")
    print("- TensorBoard logs: runs/*/")
    print("- To view TensorBoard: tensorboard --logdir runs/")

if __name__ == "__main__":
    main()
