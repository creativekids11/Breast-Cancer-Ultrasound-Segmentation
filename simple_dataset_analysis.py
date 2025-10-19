#!/usr/bin/env python3
"""
simple_dataset_analysis.py

Simple analysis of the processed BUSI dataset without visualization dependencies.
"""
import os
import pandas as pd
import cv2
import numpy as np

def load_dataset_manifest(csv_path: str) -> pd.DataFrame:
    """Load the dataset manifest CSV."""
    return pd.read_csv(csv_path)

def analyze_dataset(manifest_df: pd.DataFrame):
    """Analyze and print dataset statistics."""
    print("BUSI Dataset Analysis")
    print("=" * 50)
    
    print(f"Total samples: {len(manifest_df)}")
    print(f"Dataset: {manifest_df['dataset'].iloc[0]}")
    
    print(f"\nSplit distribution:")
    split_counts = manifest_df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} ({count/len(manifest_df)*100:.1f}%)")
    
    print(f"\nClass distribution:")
    class_counts = manifest_df['class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(manifest_df)*100:.1f}%)")
    
    print(f"\nPathology distribution:")
    pathology_counts = manifest_df['pathology'].value_counts()
    for pathology, count in pathology_counts.items():
        print(f"  {pathology}: {count} ({count/len(manifest_df)*100:.1f}%)")
    
    print(f"\nSamples with masks:")
    mask_counts = manifest_df['has_mask'].value_counts()
    for has_mask, count in mask_counts.items():
        print(f"  {has_mask}: {count} ({count/len(manifest_df)*100:.1f}%)")
    
    print(f"\nAverage number of masks per image: {manifest_df['num_masks'].mean():.2f}")
    
    # Cross-tabulation
    print(f"\nCross-tabulation (Split vs Class):")
    crosstab = pd.crosstab(manifest_df['split'], manifest_df['class'], margins=True)
    print(crosstab)
    
    return manifest_df

def check_sample_files(manifest_df: pd.DataFrame, num_samples: int = 3):
    """Check if sample files exist and get basic info."""
    print(f"\nFile Verification (checking {num_samples} samples per split):")
    print("=" * 50)
    
    for split in ['train', 'val', 'test']:
        split_data = manifest_df[manifest_df['split'] == split]
        print(f"\n{split.upper()} split:")
        
        for i, (_, row) in enumerate(split_data.head(num_samples).iterrows()):
            img_path = row['image_file_path']
            mask_path = row['roi_mask_file_path']
            
            img_exists = os.path.exists(img_path)
            mask_exists = os.path.exists(mask_path)
            
            print(f"  Sample {i+1} ({row['class']}):")
            print(f"    Image: {'✓' if img_exists else '✗'} {os.path.basename(img_path)}")
            print(f"    Mask:  {'✓' if mask_exists else '✗'} {os.path.basename(mask_path)}")
            
            if img_exists:
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"    Image shape: {img.shape}")
            
            if mask_exists:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    unique_vals = np.unique(mask)
                    print(f"    Mask shape: {mask.shape}, unique values: {unique_vals}")

def main():
    """Main function to demonstrate dataset analysis."""
    # Path to the processed dataset
    manifest_path = "./BUSI_PROCESSED/dataset_manifest.csv"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest file not found: {manifest_path}")
        print("Please run the dataset processing script first.")
        print("Example: python dataset_process.py --root './Dataset_BUSI_with_GT' --outdir './BUSI_PROCESSED' --dataset-type busi")
        return
    
    # Load manifest
    df = load_dataset_manifest(manifest_path)
    
    # Analyze dataset
    df = analyze_dataset(df)
    
    # Check sample files
    check_sample_files(df)
    
    print(f"\n" + "=" * 50)
    print("Dataset is ready for training!")
    print("=" * 50)
    print(f"You can use the processed data at: ./BUSI_PROCESSED/")
    print(f"Images are in: ./BUSI_PROCESSED/IMAGES/{{train,val,test}}/")
    print(f"Masks are in: ./BUSI_PROCESSED/MASKS/{{train,val,test}}/")
    print(f"Manifest CSV: {manifest_path}")

if __name__ == "__main__":
    main()