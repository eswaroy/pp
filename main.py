"""
Parallel Face Blurring System with Streamlit UI
A high-performance face detection and blurring application using OpenCV and multiprocessing
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import io
import os
from typing import List, Tuple, Optional
import functools
import os

# Add this after the imports
WIDER_VAL_PATH = "WIDER_val/WIDER_val/images"
DNN_MODEL_PATH = "models/opencv_face_detector_uint8.pb"
DNN_CONFIG_PATH = "models/opencv_face_detector.pbtxt"
def load_dataset_images(dataset_path: str) -> List[Tuple[str, bytes]]:
    """
    Load images from the WIDER dataset
    
    Args:
        dataset_path (str): Path to WIDER_val/images directory
        
    Returns:
        List[Tuple[str, bytes]]: List of (filename, image_data) pairs
    """
    image_data_list = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                image_data_list.append((file, image_data))
    
    return image_data_list

class FaceBlurProcessor:
    """
    High-performance face detection and blurring processor with parallel processing capabilities
    """
    
    def __init__(self, detection_method: str = "haar", blur_strength: int = 50):
        """
        Initialize the face blur processor
        
        Args:
            detection_method (str): 'haar' for Haar Cascade or 'dnn' for DNN-based detection
            blur_strength (int): Gaussian blur kernel size (must be odd)
        """
        self.detection_method = detection_method
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        
        # Initialize face detector based on method
        if detection_method == "haar":
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    raise Exception("Failed to load Haar cascade")
            except Exception as e:
                st.error(f"Error loading Haar cascade: {e}")
                st.stop()
        
        elif detection_method == "dnn":
            try:
                # Load DNN model (using OpenCV's DNN face detector)
                self.net = cv2.dnn.readNetFromTensorflow(
                    'opencv_face_detector_uint8.pb', 
                    'opencv_face_detector.pbtxt'
                )
            except Exception as e:
                st.warning(f"DNN model not found, falling back to Haar cascade: {e}")
                self.detection_method = "haar"
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar cascade classifier
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Tuple[int, int, int, int]]: List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters for speed and accuracy
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN-based detector
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Tuple[int, int, int, int]]: List of face bounding boxes (x, y, w, h)
        """
        (h, w) = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
        
        # Pass blob through network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append((x, y, x1-x, y1-y))
        
        return faces
    
    def apply_gaussian_blur(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Apply Gaussian blur to detected face regions
        
        Args:
            image (np.ndarray): Input image
            faces (List[Tuple[int, int, int, int]]): Face bounding boxes
            
        Returns:
            np.ndarray: Image with blurred faces
        """
        blurred_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = blurred_image[y:y+h, x:x+w]
            
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(face_region, (self.blur_strength, self.blur_strength), 0)
            
            # Replace original face with blurred version
            blurred_image[y:y+h, x:x+w] = blurred_face
        
        return blurred_image
    
    def process_single_image(self, image: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        Process a single image: detect faces and apply blur
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[np.ndarray, int, float]: (processed_image, face_count, processing_time)
        """
        start_time = time.time()
        
        # Detect faces based on chosen method
        if self.detection_method == "haar":
            faces = self.detect_faces_haar(image)
        else:
            faces = self.detect_faces_dnn(image)
        
        # Apply blur to detected faces
        processed_image = self.apply_gaussian_blur(image, faces)
        
        processing_time = time.time() - start_time
        
        return processed_image, len(faces), processing_time

def process_image_parallel(image_data: bytes, detection_method: str, blur_strength: int) -> Tuple[bytes, int, float]:
    """
    Wrapper function for parallel processing of images
    
    Args:
        image_data (bytes): Image data in bytes format
        detection_method (str): Detection method to use
        blur_strength (int): Blur strength
        
    Returns:
        Tuple[bytes, int, float]: (processed_image_bytes, face_count, processing_time)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Initialize processor
    processor = FaceBlurProcessor(detection_method, blur_strength)
    
    # Process image
    processed_image, face_count, processing_time = processor.process_single_image(image)
    
    # Convert back to bytes
    _, buffer = cv2.imencode('.jpg', processed_image)
    processed_bytes = buffer.tobytes()
    
    return processed_bytes, face_count, processing_time

def process_multiple_images_parallel(image_list: List[bytes], detection_method: str, 
                                   blur_strength: int, max_workers: Optional[int] = None) -> List[Tuple[bytes, int, float]]:
    """
    Process multiple images in parallel using multiprocessing
    
    Args:
        image_list (List[bytes]): List of image data in bytes format
        detection_method (str): Detection method to use
        blur_strength (int): Blur strength
        max_workers (Optional[int]): Maximum number of worker processes
        
    Returns:
        List[Tuple[bytes, int, float]]: List of (processed_image_bytes, face_count, processing_time)
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(image_list))
    
    # Create partial function with fixed parameters
    process_func = functools.partial(process_image_parallel, 
                                   detection_method=detection_method, 
                                   blur_strength=blur_strength)
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, image_list))
    
    return results

def create_streamlit_ui():
    """
    Create and configure the Streamlit user interface
    """
    st.set_page_config(
        page_title="Parallel Face Blur System",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title and description
    st.title("🎭 High-Performance Parallel Face Blurring System")
    st.markdown("""
    **Protect privacy with real-time face blurring using advanced computer vision and parallel processing.**
    
    Upload one or multiple images to automatically detect and blur faces while maintaining visual quality.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Detection method selection
    detection_method = st.sidebar.selectbox(
        "Face Detection Method",
        options=["haar", "dnn"],
        index=0,
        help="Haar: Fast but less accurate | DNN: More accurate but slower"
    )
    
    # Blur strength configuration
    blur_strength = st.sidebar.slider(
        "Blur Strength",
        min_value=11,
        max_value=99,
        value=51,
        step=2,
        help="Higher values create stronger blur effect"
    )
    
    # Parallel processing configuration
    max_workers = st.sidebar.slider(
        "Max Worker Processes",
        min_value=1,
        max_value=mp.cpu_count(),
        value=min(4, mp.cpu_count()),
        help=f"Your system has {mp.cpu_count()} CPU cores"
    )
    
    # Performance info
    st.sidebar.info(f"""
    **System Info:**
    - CPU Cores: {mp.cpu_count()}
    - Selected Workers: {max_workers}
    - Detection Method: {detection_method.upper()}
    """)
    
    return detection_method, blur_strength, max_workers

def main():
    """
    Main application function
    """
    # Initialize UI and get configuration
    detection_method, blur_strength, max_workers = create_streamlit_ui()
    process_type = st.sidebar.radio(
        "Select Input Source",
        ["Upload Files", "Use WIDER Dataset"]
    )
    
    if process_type == "Use WIDER Dataset":
        st.header("📁 WIDER Dataset Processing")
        if st.button("🚀 Process WIDER Dataset", type="primary"):
            dataset_images = load_dataset_images(WIDER_VAL_PATH)
            if dataset_images:
                st.success(f"✅ Found {len(dataset_images)} images in dataset")
                process_images(
                    [type('UploadedFile', (), {'name': name, 'read': lambda: data})() 
                     for name, data in dataset_images],
                    detection_method,
                    blur_strength,
                    max_workers
                )
            else:
                st.error("No images found in the dataset directory")
    else:
        # File upload section
        st.header("📁 Upload Images")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
            
            # Process button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Process Images", type="primary", use_container_width=True):
                    process_images(uploaded_files, detection_method, blur_strength, max_workers)
        else:
            # Show sample information
            st.info("👆 Upload one or more images to get started!")
            
            # Feature highlights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                ### ⚡ High Performance
                - Parallel processing across CPU cores
                - Optimized OpenCV algorithms
                - Real-time processing capability
                """)
            
            with col2:
                st.markdown("""
                ### 🎯 Accurate Detection
                - Haar Cascade for speed
                - DNN models for accuracy
                - Configurable detection parameters
                """)
            
            with col3:
                st.markdown("""
                ### 🎨 Quality Preservation
                - Gaussian blur for natural effect
                - Adjustable blur strength
                - Maintains image resolution
                """)

def process_images(uploaded_files, detection_method, blur_strength, max_workers):
    """
    Process uploaded images and display results
    
    Args:
        uploaded_files: Streamlit uploaded files
        detection_method (str): Detection method to use
        blur_strength (int): Blur strength
        max_workers (int): Number of worker processes
    """
    # Convert uploaded files to bytes
    image_data_list = []
    original_images = []
    filenames = []
    
    for uploaded_file in uploaded_files:
        image_data = uploaded_file.read()
        image_data_list.append(image_data)
        filenames.append(uploaded_file.name)
        
        # Store original for display
        original_images.append(Image.open(io.BytesIO(image_data)))
    
    # Processing with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    if len(uploaded_files) == 1:
        # Single image processing
        status_text.text("Processing single image...")
        processor = FaceBlurProcessor(detection_method, blur_strength)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data_list[0], np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image
        processed_image, face_count, processing_time = processor.process_single_image(cv_image)
        
        # Convert back to PIL
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_pil = Image.open(io.BytesIO(buffer.tobytes()))
        
        results = [(buffer.tobytes(), face_count, processing_time)]
        processed_images = [processed_pil]
        
    else:
        # Multiple image parallel processing
        status_text.text(f"Processing {len(uploaded_files)} images in parallel...")
        
        results = process_multiple_images_parallel(
            image_data_list, detection_method, blur_strength, max_workers
        )
        
        # Convert results to PIL images
        processed_images = []
        for result_bytes, _, _ in results:
            processed_pil = Image.open(io.BytesIO(result_bytes))
            processed_images.append(processed_pil)
    
    total_time = time.time() - start_time
    progress_bar.progress(1.0)
    
    # Display results
    st.header("🎯 Processing Results")
    
    # Summary statistics
    total_faces = sum(result[1] for result in results)
    avg_processing_time = sum(result[2] for result in results) / len(results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Images Processed", len(uploaded_files))
    with col2:
        st.metric("Total Faces Detected", total_faces)
    with col3:
        st.metric("Avg Processing Time", f"{avg_processing_time:.3f}s")
    with col4:
        st.metric("Total Time", f"{total_time:.3f}s")
    
    # Performance metrics
    if len(uploaded_files) > 1:
        speedup = (avg_processing_time * len(uploaded_files)) / total_time
        st.info(f"🚀 Parallel processing achieved {speedup:.2f}x speedup!")
    
    # Display images side by side
    st.header("📊 Before & After Comparison")
    
    for i, (original, processed, filename, (_, face_count, proc_time)) in enumerate(
        zip(original_images, processed_images, filenames, results)
    ):
        st.subheader(f"Image {i+1}: {filename}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(original, use_container_width=True)
        
        with col2:
            st.markdown("**Blurred Image**")
            st.image(processed, use_container_width=True)
        
        # Image-specific metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Faces Detected", face_count)
        with col2:
            st.metric("Processing Time", f"{proc_time:.3f}s")
        with col3:
            # Download button for processed image
            img_buffer = io.BytesIO()
            processed.save(img_buffer, format="JPEG")
            st.download_button(
                label="💾 Download",
                data=img_buffer.getvalue(),
                file_name=f"blurred_{filename}",
                mime="image/jpeg"
            )
        
        st.divider()
    
    status_text.success("✅ Processing completed successfully!")

if __name__ == "__main__":
    main()