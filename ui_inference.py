#!/usr/bin/env python3
"""
Breast Cancer Ultrasound Segmentation - Inference UI
A Streamlit web application for running inference on ultrasound images
"""
import streamlit as st
# import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

# Import our inference functionality
from simple_inference import load_trained_model, predict

# Configure page
st.set_page_config(
    page_title="Breast Cancer Ultrasound Segmentation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)"""
    with st.spinner("Loading trained model..."):
        model = load_trained_model()
    return model

def get_available_images():
    """Get list of available images from BUSI_PROCESSED"""
    base_path = Path("BUSI_PROCESSED/IMAGES")

    if not base_path.exists():
        st.error(f"Directory {base_path} not found!")
        return [], [], []

    images = []
    categories = []
    paths = []

    # Scan all subdirectories
    for category_dir in base_path.iterdir():
        if category_dir.is_dir():
            category = category_dir.name
            for img_file in category_dir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                    images.append(f"{category} - {img_file.name}")
                    categories.append(category)
                    paths.append(str(img_file))

    return images, categories, paths

def display_image_comparison(original_path, result):
    """Display original image, mask, and probability map side by side"""
    col1, col2, col3 = st.columns(3)

    # Original image
    with col1:
        st.subheader("📸 Original Image")
        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        st.image(original_img, use_column_width=True, caption="Ultrasound Image")

    # Segmentation mask
    with col2:
        st.subheader("🎯 Segmentation Mask")
        mask = result['mask']
        # Convert binary mask to RGB for better visualization
        mask_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # Add red overlay
        mask_rgb[mask == 1] = [255, 0, 0]  # Red for tumor
        st.image(mask_rgb, use_column_width=True, caption="Predicted Tumor Region")

    # Probability map
    with col3:
        st.subheader("🔥 Probability Map")
        prob_map = result['probabilities']
        # Create heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        ax.set_title("Tumor Probability")
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)

def calculate_tumor_stats(result):
    """Calculate tumor statistics from prediction"""
    mask = result['mask']
    probabilities = result['probabilities']

    # Basic statistics
    total_pixels = mask.size
    tumor_pixels = np.sum(mask)
    tumor_percentage = (tumor_pixels / total_pixels) * 100

    # Confidence score (mean probability in tumor region)
    tumor_probabilities = probabilities[mask == 1]
    mean_confidence = np.mean(tumor_probabilities) if len(tumor_probabilities) > 0 else 0
    max_confidence = np.max(tumor_probabilities) if len(tumor_probabilities) > 0 else 0

    return {
        'tumor_pixels': int(tumor_pixels),
        'total_pixels': total_pixels,
        'tumor_percentage': tumor_percentage,
        'mean_confidence': mean_confidence,
        'max_confidence': max_confidence
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Breast Cancer Ultrasound Segmentation</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("🔧 Controls")

    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load trained model!")
        return

    st.sidebar.success("✅ Model loaded successfully!")

    # Get available images
    with st.sidebar:
        with st.spinner("Scanning for images..."):
            image_names, categories, image_paths = get_available_images()

    if not image_names:
        st.error("❌ No images found in BUSI_PROCESSED/IMAGES/")
        st.info("Make sure you have processed the BUSI dataset first.")
        return

    # Image selection
    st.sidebar.subheader("📁 Select Images")

    # Category filter
    unique_categories = list(set(categories))
    selected_categories = st.sidebar.multiselect(
        "Filter by category:",
        unique_categories,
        default=unique_categories,
        help="Select which image categories to show"
    )

    # Filter images by category
    filtered_indices = [i for i, cat in enumerate(categories) if cat in selected_categories]
    filtered_names = [image_names[i] for i in filtered_indices]
    filtered_paths = [image_paths[i] for i in filtered_indices]

    # Image selection mode
    mode = st.sidebar.radio(
        "Selection Mode:",
        ["Single Image", "Batch Processing"],
        help="Choose how to process images"
    )

    if mode == "Single Image":
        # Single image selection
        selected_image = st.sidebar.selectbox(
            "Select an image:",
            filtered_names,
            help="Choose an image to analyze"
        )

        if selected_image:
            image_idx = filtered_names.index(selected_image)
            selected_path = filtered_paths[image_idx]

            # Inference button
            if st.sidebar.button("🔍 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Running inference..."):
                    start_time = time.time()
                    result = predict(model, selected_path, threshold=0.5)
                    inference_time = time.time() - start_time

                # Display results
                st.success(f"✅ Analysis completed in {inference_time:.2f} seconds!")

                # Statistics
                stats = calculate_tumor_stats(result)

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tumor Pixels", f"{stats['tumor_pixels']:,}")
                with col2:
                    st.metric("Tumor Area", f"{stats['tumor_percentage']:.1f}%")
                with col3:
                    st.metric("Mean Confidence", f"{stats['mean_confidence']:.3f}")
                with col4:
                    st.metric("Max Confidence", f"{stats['max_confidence']:.3f}")

                # Image comparison
                st.subheader("📊 Analysis Results")
                display_image_comparison(selected_path, result)

                # Additional info
                with st.expander("📋 Detailed Information"):
                    st.write(f"**Image Path:** {selected_path}")
                    st.write(f"**Image Size:** {result['original_size']}")
                    st.write(f"**Inference Time:** {inference_time:.2f} seconds")
                    st.write(f"**Threshold Used:** 0.5")

                    # Download options
                    st.subheader("💾 Download Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Convert mask to PIL Image for download
                        mask_img = Image.fromarray((result['mask'] * 255).astype(np.uint8))
                        st.download_button(
                            label="📥 Download Mask",
                            data=cv2.imencode('.png', (result['mask'] * 255).astype(np.uint8))[1].tobytes(),
                            file_name=f"{Path(selected_path).stem}_mask.png",
                            mime="image/png"
                        )

                    with col2:
                        # Save probabilities as numpy array
                        prob_bytes = result['probabilities'].tobytes()
                        st.download_button(
                            label="📥 Download Probabilities",
                            data=prob_bytes,
                            file_name=f"{Path(selected_path).stem}_probabilities.npy",
                            mime="application/octet-stream"
                        )

    else:  # Batch Processing
        st.sidebar.subheader("📦 Batch Processing")

        # Select multiple images
        selected_images = st.sidebar.multiselect(
            "Select images for batch processing:",
            filtered_names,
            max_selections=10,  # Limit for performance
            help="Select up to 10 images for batch analysis"
        )

        if selected_images:
            st.sidebar.info(f"Selected {len(selected_images)} images")

            # Batch inference button
            if st.sidebar.button("🚀 Process Batch", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(selected_images)} images..."):
                    start_time = time.time()

                    # Process all selected images
                    batch_results = []
                    for img_name in selected_images:
                        img_idx = filtered_names.index(img_name)
                        img_path = filtered_paths[img_idx]

                        result = predict(model, img_path, threshold=0.5)
                        stats = calculate_tumor_stats(result)

                        batch_results.append({
                            'image': img_name,
                            'path': img_path,
                            'tumor_percentage': stats['tumor_percentage'],
                            'mean_confidence': stats['mean_confidence'],
                            'max_confidence': stats['max_confidence']
                        })

                    total_time = time.time() - start_time

                # Display batch results
                st.success(f"✅ Batch processing completed in {total_time:.2f} seconds!")

                # Summary statistics
                st.subheader("📊 Batch Summary")
                df = pd.DataFrame(batch_results)
                st.dataframe(df)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Images Processed", len(batch_results))
                with col2:
                    st.metric("Avg Tumor Area", f"{df['tumor_percentage'].mean():.1f}%")
                with col3:
                    st.metric("Avg Confidence", f"{df['mean_confidence'].mean():.3f}")

                # Individual results
                st.subheader("🔍 Individual Results")
                for result in batch_results:
                    with st.expander(f"📸 {result['image']}"):
                        # Quick preview
                        img_result = predict(model, result['path'], threshold=0.5)
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            orig_img = cv2.imread(result['path'], cv2.IMREAD_GRAYSCALE)
                            st.image(orig_img, caption="Original", width=150)

                        with col2:
                            mask_rgb = cv2.cvtColor((img_result['mask'] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                            mask_rgb[img_result['mask'] == 1] = [255, 0, 0]
                            st.image(mask_rgb, caption="Mask", width=150)

                        with col3:
                            fig, ax = plt.subplots(figsize=(3, 3))
                            ax.imshow(img_result['probabilities'], cmap='hot', vmin=0, vmax=1)
                            ax.axis('off')
                            st.pyplot(fig)

                        # Stats
                        st.write(f"**Tumor Area:** {result['tumor_percentage']:.1f}%")
                        st.write(f"**Confidence:** {result['mean_confidence']:.3f}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Breast Cancer Ultrasound Segmentation System</p>
        <p>Powered by Gradient Boosting with Pre-trained Models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()