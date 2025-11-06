"""
Real-Time Personal Protective Equipment (PPE) Detector
Week 4 Capstone Project - Vision AI Training Program

This application detects whether a person in a video feed is wearing:
- Hard Hat (Helmet)
- Safety Vest (Jacket)

Features:
- Real-time webcam detection
- Video file processing
- Frame-by-frame inference with YOLO model
- Bounding boxes and labels visualization
- Compliance status display
"""

import cv2
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# Working directory ayarla
WORK_DIR = Path(__file__).parent
import os
os.chdir(WORK_DIR)


class PPEDetector:
    """Personal Protective Equipment Detector using YOLO."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize PPE Detector.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_names = {}
        
        # PPE classes we're looking for
        self.target_classes = {
            'helmet': ['helmet', 'hardhat', 'hard hat'],
            'vest': ['jacket', 'vest', 'safety vest', 'safety jacket']
        }
        
        self.load_model()
    
    def load_model(self):
        """Load YOLO model."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            print(f"📦 Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"✅ Model loaded successfully!")
                print(f"   Classes: {list(self.class_names.values())}")
            else:
                raise ValueError("Model does not have class names")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def check_class_match(self, detected_class: str, target_keywords: list) -> bool:
        """
        Check if detected class matches target keywords.
        
        Args:
            detected_class: Detected class name
            target_keywords: List of keywords to match
            
        Returns:
            True if match found
        """
        detected_lower = detected_class.lower()
        return any(keyword.lower() in detected_lower for keyword in target_keywords)
    
    def detect_ppe(self, frame):
        """
        Detect PPE in a frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Tuple of (annotated_frame, detection_info)
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        result = results[0]
        
        # Initialize detection info
        detection_info = {
            'helmet_detected': False,
            'vest_detected': False,
            'helmet_boxes': [],
            'vest_boxes': [],
            'all_detections': []
        }
        
        # Process detections
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[class_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Check if it's a helmet
                if self.check_class_match(class_name, self.target_classes['helmet']):
                    detection_info['helmet_detected'] = True
                    detection_info['helmet_boxes'].append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class': class_name
                    })
                
                # Check if it's a safety vest
                if self.check_class_match(class_name, self.target_classes['vest']):
                    detection_info['vest_detected'] = True
                    detection_info['vest_boxes'].append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class': class_name
                    })
                
                detection_info['all_detections'].append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
        
        # Draw annotations
        annotated_frame = self.draw_detections(frame.copy(), detection_info)
        
        return annotated_frame, detection_info
    
    def draw_detections(self, frame, detection_info):
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            detection_info: Detection information dictionary
            
        Returns:
            Annotated frame
        """
        # Draw helmet detections (green if detected, red if not)
        if detection_info['helmet_detected']:
            for det in detection_info['helmet_boxes']:
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                class_name = det['class']
                
                # Green box for helmet detected
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Helmet: {class_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Show warning if no helmet detected
            cv2.putText(frame, "NO HELMET DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw vest detections (green if detected, red if not)
        if detection_info['vest_detected']:
            for det in detection_info['vest_boxes']:
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                class_name = det['class']
                
                # Green box for vest detected
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Vest: {class_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Show warning if no vest detected
            cv2.putText(frame, "NO VEST DETECTED", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw compliance status
        helmet_status = "✓" if detection_info['helmet_detected'] else "✗"
        vest_status = "✓" if detection_info['vest_detected'] else "✗"
        
        status_text = f"Helmet: {helmet_status} | Vest: {vest_status}"
        status_color = (0, 255, 0) if (detection_info['helmet_detected'] and 
                                       detection_info['vest_detected']) else (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame
    
    def process_webcam(self):
        """Process video feed from webcam."""
        print("\n" + "=" * 60)
        print("Real-Time PPE Detection - Webcam Mode")
        print("Press 'q' to quit")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n✅ Webcam opened successfully")
        print("   Processing frames...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from webcam")
                break
            
            # Detect PPE
            annotated_frame, detection_info = self.detect_ppe(frame)
            
            # Display frame
            cv2.imshow('PPE Detector - Real-Time', annotated_frame)
            
            frame_count += 1
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                helmet = "✓" if detection_info['helmet_detected'] else "✗"
                vest = "✓" if detection_info['vest_detected'] else "✗"
                print(f"Frame {frame_count}: Helmet: {helmet} | Vest: {vest}")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Webcam processing completed")
    
    def process_image(self, image_path: str, output_path: str = None):
        """
        Process single image.
        
        Args:
            image_path: Path to input image file
            output_path: Path to save output image (optional)
        """
        print("\n" + "=" * 60)
        print("PPE Detection - Image Mode")
        print("=" * 60)
        
        if not Path(image_path).exists():
            print(f"❌ Error: Image file not found: {image_path}")
            return
        
        print(f"\n📸 Image: {image_path}")
        print("🔍 Processing image...")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return
        
        # Detect PPE
        annotated_frame, detection_info = self.detect_ppe(frame)
        
        # Save or display
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"✅ Result saved to: {output_path}")
        else:
            # Display image
            cv2.imshow('PPE Detector - Image', annotated_frame)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print detection summary
        helmet = "✓" if detection_info['helmet_detected'] else "✗"
        vest = "✓" if detection_info['vest_detected'] else "✗"
        print(f"\n📊 Detection Summary:")
        print(f"   Helmet: {helmet}")
        print(f"   Vest: {vest}")
        if detection_info['all_detections']:
            print(f"   Total detections: {len(detection_info['all_detections'])}")
    
    def process_video(self, video_path: str, output_path: str = None):
        """
        Process video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
        """
        print("\n" + "=" * 60)
        print("Real-Time PPE Detection - Video Mode")
        print("=" * 60)
        
        # Tırnak işaretlerini temizle
        video_path = video_path.strip('"').strip("'").strip()
        
        # Path kontrolü
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            # Relative path denemesi
            if not video_path_obj.is_absolute():
                video_path_obj = WORK_DIR / video_path
            
            if not video_path_obj.exists():
                print(f"❌ Error: Video file not found: {video_path}")
                return
        
        video_path = str(video_path_obj.absolute())
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video properties:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"   Output: {output_path}")
        
        print("\n⏳ Processing video...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect PPE
            annotated_frame, detection_info = self.detect_ppe(frame)
            
            # Write frame if output specified
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('PPE Detector - Video', annotated_frame)
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                helmet = "✓" if detection_info['helmet_detected'] else "✗"
                vest = "✓" if detection_info['vest_detected'] else "✗"
                print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                      f"Helmet: {helmet} | Vest: {vest}")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Video processing completed")
        print(f"   Processed {frame_count} frames")
        if output_path:
            print(f"   Output saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Real-Time Personal Protective Equipment (PPE) Detector'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='runs/train/safety_equipment/weights/best.pt',
        help='Path to trained YOLO model'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['webcam', 'video', 'image'],
        help='Detection mode: webcam, video, or image (optional - will show menu if not provided)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input video/image file (required for video/image mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save output video/image (optional)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found: {args.model}")
        print(f"   Please train the model first: python train_model.py")
        sys.exit(1)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🎮 GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU not available, using CPU (may be slower)")
    
    # Initialize detector
    detector = PPEDetector(model_path=args.model, conf_threshold=args.conf)
    
    # If mode is provided via command line, use it directly
    if args.mode:
        if args.mode == 'webcam':
            detector.process_webcam()
        elif args.mode == 'video':
            if not args.input:
                print("❌ Error: --input required for video mode")
                sys.exit(1)
            detector.process_video(args.input, args.output)
        elif args.mode == 'image':
            if not args.input:
                print("❌ Error: --input required for image mode")
                sys.exit(1)
            detector.process_image(args.input, args.output)
    else:
        # Interactive menu mode
        print("\n" + "=" * 60)
        print("Real-Time Personal Protective Equipment (PPE) Detector")
        print("=" * 60)
        print(f"\n📦 Model: {args.model}")
        print(f"🎯 Confidence threshold: {args.conf}")
        
        print("\nNe yapmak istersiniz?")
        print("1. Webcam ile real-time detection")
        print("2. Video dosyası üzerinde detection")
        print("3. Tek görüntü üzerinde detection")
        
        choice = input("\nSeçiminiz (1-3): ").strip()
        
        if choice == "1":
            detector.process_webcam()
        elif choice == "2":
            video_path = input("\nVideo path'i girin: ").strip()
            save_output = input("Sonucu kaydetmek istiyor musunuz? (e/h, varsayılan: h): ").strip().lower()
            output_path = None
            if save_output == 'e' or save_output == 'evet' or save_output == 'y' or save_output == 'yes':
                output_path = input("Output path'i girin (örn: output.mp4): ").strip()
                if not output_path:
                    output_path = "output.mp4"
            detector.process_video(video_path, output_path)
        elif choice == "3":
            image_path = input("\nGörüntü path'i girin: ").strip()
            save_output = input("Sonucu kaydetmek istiyor musunuz? (e/h, varsayılan: h): ").strip().lower()
            output_path = None
            if save_output == 'e' or save_output == 'evet' or save_output == 'y' or save_output == 'yes':
                output_path = input("Output path'i girin (örn: output.jpg): ").strip()
                if not output_path:
                    output_path = "output.jpg"
            detector.process_image(image_path, output_path)
        else:
            print("❌ Geçersiz seçim!")
            sys.exit(1)


if __name__ == '__main__':
    main()

