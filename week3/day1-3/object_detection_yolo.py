"""
Week 3 - Days 1-3: Object Detection
Understanding bounding boxes, IoU, NMS, and using YOLO for real-time detection
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
import os


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, 
                       iou_threshold: float = 0.5) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping boxes.
    
    Args:
        boxes: Array of bounding boxes [N, 4] in [x1, y1, x2, y2] format
        scores: Array of confidence scores [N]
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    # Sort boxes by score (descending)
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        # Keep the box with highest score
        current = indices[0]
        keep.append(current)
        
        # Remove boxes with high IoU
        remaining = indices[1:]
        if len(remaining) == 0:
            break
        
        ious = np.array([calculate_iou(boxes[current], boxes[i]) 
                        for i in remaining])
        indices = remaining[ious <= iou_threshold]
    
    return np.array(keep)


def demonstrate_iou_and_nms():
    """Demonstrate IoU calculation and NMS."""
    print("=" * 60)
    print("IoU and NMS Demonstration")
    print("=" * 60)
    
    # Create sample boxes
    boxes = np.array([
        [100, 100, 200, 200],  # Box 1
        [110, 110, 210, 210],  # Box 2 (overlaps with Box 1)
        [300, 300, 400, 400],  # Box 3 (no overlap)
        [105, 105, 205, 205],  # Box 4 (overlaps with Box 1)
    ])
    
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    
    # Calculate IoU between Box 1 and others
    print("\nIoU between Box 1 and other boxes:")
    for i in range(1, len(boxes)):
        iou = calculate_iou(boxes[0], boxes[i])
        print(f"  Box 1 vs Box {i + 1}: {iou:.3f}")
    
    # Apply NMS
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
    print(f"\nNMS Results (IoU threshold=0.5):")
    print(f"  Original boxes: {len(boxes)}")
    print(f"  Remaining boxes: {len(keep_indices)}")
    print(f"  Kept indices: {keep_indices}")
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (box, score, color) in enumerate(zip(boxes, scores, colors)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        alpha = 0.3 if i in keep_indices else 0.1
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor=color, 
                        facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'Box {i+1} ({score:.2f})', 
               fontsize=10, color=color, weight='bold')
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect('equal')
    ax.set_title('IoU and NMS Visualization')
    ax.grid(True, alpha=0.3)
    
    output_dir = Path('week3/day1-3/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'iou_nms_demo.png')
    print(f"\nVisualization saved to {output_dir / 'iou_nms_demo.png'}")


def detect_objects_image(model: YOLO, image_path: str, save_path: str = None):
    """Detect objects in an image using YOLO."""
    # Run inference
    results = model(image_path)
    
    # Get first result
    result = results[0]
    
    # Draw bounding boxes
    annotated_image = result.plot()
    
    # Save result
    if save_path:
        cv2.imwrite(save_path, annotated_image)
        print(f"Result saved to {save_path}")
    
    # Print detections
    print(f"\nDetected {len(result.boxes)} objects:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"  - {class_name}: {confidence:.2f}")
    
    return annotated_image, result


def detect_objects_webcam(model: YOLO):
    """Real-time object detection using webcam."""
    print("\n" + "=" * 60)
    print("Real-time Object Detection (Webcam)")
    print("Press 'q' to quit")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠ Webcam not available")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Display
        cv2.imshow('YOLO Object Detection', annotated_frame)
        
        frame_count += 1
        if frame_count >= 300:  # Run for ~10 seconds at 30fps
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection completed")


def demonstrate_yolo_detection():
    """Demonstrate YOLO object detection."""
    print("\n" + "=" * 60)
    print("YOLO Object Detection")
    print("=" * 60)
    
    # Load YOLO model
    print("\nLoading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # nano version (fastest)
    print("✓ Model loaded")
    
    # Create sample image if doesn't exist
    output_dir = Path('week3/day1-3/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(test_image, (450, 200), 50, (255, 0, 0), -1)
    cv2.putText(test_image, 'Test Image', (200, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    test_image_path = str(output_dir / 'test_image.jpg')
    cv2.imwrite(test_image_path, test_image)
    
    # Detect objects in image
    print("\nDetecting objects in test image...")
    result_image, results = detect_objects_image(
        model, test_image_path, 
        str(output_dir / 'detection_result.jpg')
    )
    
    # Optional: Webcam detection
    # print("\nStarting webcam detection...")
    # detect_objects_webcam(model)


def demonstrate_yolo_video():
    """Demonstrate YOLO on video file."""
    print("\n" + "=" * 60)
    print("YOLO Video Detection")
    print("=" * 60)
    
    model = YOLO('yolov8n.pt')
    
    # You can process a video file like this:
    # video_path = "path/to/your/video.mp4"
    # results = model(video_path)
    # results[0].save("output_video.mp4")
    
    print("Video processing example (commented out)")
    print("Uncomment and provide video path to test")


if __name__ == "__main__":
    # Demonstrate IoU and NMS
    demonstrate_iou_and_nms()
    
    # Demonstrate YOLO
    demonstrate_yolo_detection()
    
    # Optional: Webcam detection
    # demonstrate_yolo_detection()  # Uncomment to test webcam
    
    print("\n" + "=" * 60)
    print("Object detection demonstrations completed!")
    print("=" * 60)

