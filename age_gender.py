import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from collections import deque, defaultdict
import csv
import face_recognition
from datetime import datetime
import mediapipe as mp

def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou_val

class ImprovedGenderAgeDetector:
    def __init__(self):
        """Initialize the improved gender and age detection model"""
        # Define the model files
        self.face_proto = "opencv_face_detector.pbtxt"
        self.face_model = "opencv_face_detector_uint8.pb"
        self.age_proto = "age_deploy.prototxt"
        self.age_model = "age_net.caffemodel"
        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "gender_net.caffemodel"
        
        # Load networks
        self.face_net = cv2.dnn.readNet(self.face_model, self.face_proto)
        self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)
        self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
        
        # Define age and gender labels
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                        '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        
        # Model parameters
        self.model_mean = (78.4263377603, 87.7689143744, 114.895847746)
        
        # MediaPipe face detection with improved settings
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,  # Use model 1 for better accuracy
            min_detection_confidence=0.7  # Higher confidence threshold
        )
        
        # Haar cascade for additional face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Confidence thresholds
        self.min_face_confidence = 0.7
        self.min_age_confidence = 0.6
        self.min_gender_confidence = 0.7
        
        # Face preprocessing parameters
        self.face_size = (227, 227)
        self.histogram_equalization = True
        self.gaussian_blur = True
        
        # Ensemble parameters
        self.use_ensemble = True
        self.ensemble_weight_mediapipe = 0.6
        self.ensemble_weight_haar = 0.3
        self.ensemble_weight_dnn = 0.1

    def preprocess_face(self, face_img):
        """Apply preprocessing to improve detection accuracy with lighting compensation"""
        if face_img is None or face_img.size == 0:
            return None
            
        # Resize to standard size
        face_img = cv2.resize(face_img, self.face_size)
        
        # Apply lighting compensation for better accuracy
        face_img = self.enhance_lighting_for_prediction(face_img)
        
        # Apply histogram equalization for better contrast
        if self.histogram_equalization:
            # Convert to LAB color space for better histogram equalization
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply Gaussian blur to reduce noise
        if self.gaussian_blur:
            face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
        
        return face_img
    
    def enhance_lighting_for_prediction(self, face_img):
        """Enhance lighting specifically for age/gender prediction accuracy"""
        if face_img is None or face_img.size == 0:
            return face_img
            
        # Convert to LAB color space for better lighting compensation
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Analyze lighting conditions and apply appropriate correction
        mean_intensity = np.mean(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY))
        
        # Apply gamma correction based on lighting conditions
        if mean_intensity < 60:  # Dark image
            gamma = 1.3
        elif mean_intensity > 180:  # Bright image
            gamma = 0.8
        else:  # Normal lighting
            gamma = 1.0
        
        if gamma != 1.0:
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_bgr = cv2.LUT(enhanced_bgr, table)
        
        # Additional contrast enhancement for better feature extraction
        # Convert to float for better precision
        img_float = enhanced_bgr.astype(np.float32) / 255.0
        
        # Enhance contrast
        contrast_factor = 1.2
        brightness_factor = 0.1
        
        # Apply contrast and brightness adjustment
        enhanced_float = img_float * contrast_factor + brightness_factor
        enhanced_float = np.clip(enhanced_float, 0, 1)
        
        # Convert back to uint8
        enhanced_bgr = (enhanced_float * 255).astype(np.uint8)
        
        return enhanced_bgr
    
    def get_face_boxes_mediapipe(self, frame):
        """Get face boxes using MediaPipe with lighting enhancement"""
        # Enhance lighting for better face detection
        enhanced_frame = self.enhance_lighting_for_prediction(frame)
        
        h, w = enhanced_frame.shape[:2]
        results = self.face_detector.process(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        face_boxes = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] >= self.min_face_confidence:
                    bboxC = detection.location_data.relative_bounding_box
                    x1 = int(bboxC.xmin * w)
                    y1 = int(bboxC.ymin * h)
                    x2 = int((bboxC.xmin + bboxC.width) * w)
                    y2 = int((bboxC.ymin + bboxC.height) * h)
                    # Clamp to image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w-1, x2)
                    y2 = min(h-1, y2)
                    face_boxes.append([x1, y1, x2, y2])
        return face_boxes

    def get_face_boxes_haar(self, frame):
        """Get face boxes using Haar Cascade with lighting enhancement"""
        # Enhance lighting for better detection
        enhanced_frame = self.enhance_lighting_for_prediction(frame)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection parameters for different lighting conditions
        faces = []
        
        # Method 1: Standard detection
        faces1 = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        faces.extend(faces1)
        
        # Method 2: Enhanced detection for low light
        if len(faces) == 0:
            # Apply additional histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            faces2 = self.face_cascade.detectMultiScale(
                enhanced_gray, 
                scaleFactor=1.05, 
                minNeighbors=3, 
                minSize=(25, 25)
            )
            faces.extend(faces2)
        
        # Method 3: Adaptive threshold detection for very low light
        if len(faces) == 0:
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            faces3 = self.face_cascade.detectMultiScale(
                adaptive_thresh, 
                scaleFactor=1.05, 
                minNeighbors=2, 
                minSize=(20, 20)
            )
            faces.extend(faces3)
        
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append([x, y, x+w, y+h])
        return face_boxes

    def get_face_boxes_dnn(self, frame):
        """Get face boxes using DNN face detector with lighting enhancement"""
        # Enhance lighting for better detection
        enhanced_frame = self.enhance_lighting_for_prediction(frame)
        
        h, w = enhanced_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(enhanced_frame, 1.0, (300, 300), (104, 177, 123), False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        face_boxes = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_face_confidence:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w-1, x2)
                y2 = min(h-1, y2)
                face_boxes.append([x1, y1, x2, y2])
        return face_boxes

    def get_face_box(self, frame):
        """Get face boxes using ensemble of detection methods"""
        if not self.use_ensemble:
            return self.get_face_boxes_mediapipe(frame)
        
        # Get faces from all methods
        mediapipe_faces = self.get_face_boxes_mediapipe(frame)
        haar_faces = self.get_face_boxes_haar(frame)
        dnn_faces = self.get_face_boxes_dnn(frame)
        
        # Combine and deduplicate faces
        all_faces = []
        
        # Add MediaPipe faces with weight
        for face in mediapipe_faces:
            all_faces.append((face, self.ensemble_weight_mediapipe))
        
        # Add Haar faces with weight
        for face in haar_faces:
            all_faces.append((face, self.ensemble_weight_haar))
        
        # Add DNN faces with weight
        for face in dnn_faces:
            all_faces.append((face, self.ensemble_weight_dnn))
        
        # Merge overlapping faces
        merged_faces = self.merge_overlapping_faces(all_faces)
        return merged_faces

    def merge_overlapping_faces(self, weighted_faces):
        """Merge overlapping face detections"""
        if not weighted_faces:
            return []
        
        # Sort by confidence (weight)
        weighted_faces.sort(key=lambda x: x[1], reverse=True)
        
        merged = []
        used = set()
        
        for i, (face1, weight1) in enumerate(weighted_faces):
            if i in used:
                continue
                
            merged_face = list(face1)
            total_weight = weight1
            used.add(i)
            
            # Check for overlaps with remaining faces
            for j, (face2, weight2) in enumerate(weighted_faces[i+1:], i+1):
                if j in used:
                    continue
                    
                if iou(face1, face2) > 0.3:  # Overlap threshold
                    # Merge boxes using weighted average
                    merged_face[0] = (merged_face[0] * total_weight + face2[0] * weight2) / (total_weight + weight2)
                    merged_face[1] = (merged_face[1] * total_weight + face2[1] * weight2) / (total_weight + weight2)
                    merged_face[2] = (merged_face[2] * total_weight + face2[2] * weight2) / (total_weight + weight2)
                    merged_face[3] = (merged_face[3] * total_weight + face2[3] * weight2) / (total_weight + weight2)
                    total_weight += weight2
                    used.add(j)
            
            merged.append([int(x) for x in merged_face])
        
        return merged

    def predict_age_gender(self, face):
        """Predict age and gender from face ROI with improved preprocessing and lighting compensation"""
        # Try multiple lighting enhancement methods for better accuracy
        enhanced_versions = []
        
        # Version 1: Standard enhancement
        enhanced_versions.append(self.enhance_lighting_for_prediction(face))
        
        # Version 2: Additional contrast enhancement
        enhanced_face2 = self.enhance_lighting_for_prediction(face)
        # Apply additional contrast enhancement
        lab = cv2.cvtColor(enhanced_face2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced_versions.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        
        # Version 3: Normalized lighting
        enhanced_face3 = face.copy()
        # Normalize lighting
        img_float = enhanced_face3.astype(np.float32) / 255.0
        mean_val = np.mean(img_float)
        std_val = np.std(img_float)
        if std_val > 0:
            normalized = (img_float - mean_val) / std_val * 0.2 + 0.5
            normalized = np.clip(normalized, 0, 1)
            enhanced_versions.append((normalized * 255).astype(np.uint8))
        else:
            enhanced_versions.append(enhanced_face3)
        
        # Try all enhanced versions and select the best prediction
        best_result = None
        best_confidence = 0
        
        for enhanced_face in enhanced_versions:
            # Preprocess face
            processed_face = self.preprocess_face(enhanced_face)
            if processed_face is None:
                continue
            
            blob = cv2.dnn.blobFromImage(processed_face, 1.0, (227, 227), self.model_mean, swapRB=False)
            
            # Predict gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_confidence = gender_preds[0].max()
            
            if gender_confidence < self.min_gender_confidence:
                continue
                
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_confidence = age_preds[0].max()
            
            if age_confidence < self.min_age_confidence:
                continue
                
            age = self.age_list[age_preds[0].argmax()]
            
            # Calculate combined confidence
            combined_confidence = (gender_confidence + age_confidence) / 2
            
            # Keep the best result
            if combined_confidence > best_confidence:
                best_confidence = combined_confidence
                best_result = (gender, age, gender_confidence, age_confidence)
        
        # If no enhanced version worked, try original with standard preprocessing
        if best_result is None:
            processed_face = self.preprocess_face(face)
            if processed_face is None:
                return None, None, 0, 0
            
            blob = cv2.dnn.blobFromImage(processed_face, 1.0, (227, 227), self.model_mean, swapRB=False)
            
            # Predict gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_confidence = gender_preds[0].max()
            
            if gender_confidence < self.min_gender_confidence:
                return None, None, 0, 0
                
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_confidence = age_preds[0].max()
            
            if age_confidence < self.min_age_confidence:
                return None, None, 0, 0
                
            age = self.age_list[age_preds[0].argmax()]
            
            return gender, age, gender_confidence, age_confidence
        
        return best_result

    def process_image(self, image_path):
        """Process a single image with improved detection"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not load image: {image_path}")
            return None
            
        face_boxes = self.get_face_box(frame)
        log_file = 'age_gender_log.csv'
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['face_id', 'gender', 'age', 'confidence', 'bbox', 'source', 'timestamp'])
        
        logged_encodings = []
        SIMILARITY_THRESHOLD = 0.6
        
        for i, face_box in enumerate(face_boxes):
            # Extract face with padding
            face = frame[max(0, face_box[1]-20):min(face_box[3]+20, frame.shape[0]-1),
                        max(0, face_box[0]-20):min(face_box[2]+20, frame.shape[1]-1)]
            
            # Compute face encoding for uniqueness
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            
            is_new = True
            if encodings:
                encoding = encodings[0]
                for logged in logged_encodings:
                    dist = np.linalg.norm(encoding - logged)
                    if dist < SIMILARITY_THRESHOLD:
                        is_new = False
                        break
                
                if is_new:
                    result = self.predict_age_gender(face)
                    if result[0] is not None:  # Only log if prediction is successful
                        gender, age, gender_conf, age_conf = result
                        avg_confidence = (gender_conf + age_conf) / 2
                        
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([i, gender, age, f"{avg_confidence:.3f}", 
                                           f'{face_box[0]},{face_box[1]},{face_box[2]},{face_box[3]}', 
                                           image_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                        logged_encodings.append(encoding)
            
            # Draw rectangle around face (blue)
            cv2.rectangle(frame, (face_box[0], face_box[1]), 
                         (face_box[2], face_box[3]), (60, 120, 255), 2)
            
            # Add labels with confidence
            result = self.predict_age_gender(face)
            if result[0] is not None:
                gender, age, gender_conf, age_conf = result
                avg_confidence = (gender_conf + age_conf) / 2
                label = f"{gender} | {age} | {avg_confidence:.2f}"
            else:
                label = "Detection Failed"
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            overlay = frame.copy()
            y1 = max(face_box[1] - label_size[1] - 16, 0)
            y2 = face_box[1]
            x1 = face_box[0]
            x2 = face_box[0] + label_size[0] + 16
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, label, (x1 + 8, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

    def process_video(self, video_source=0):
        """Process video stream with improved detection"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return
            
        print("Press 'q' to quit, 's' to save current frame")
        frame_count = 0
        log_file = 'age_gender_log.csv'
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['face_id', 'gender', 'age', 'confidence', 'bbox', 'frame', 'timestamp'])
        
        # Use a buffer for each face (by bbox) to smooth age
        age_buffers = defaultdict(lambda: deque(maxlen=15))
        gender_buffers = defaultdict(lambda: deque(maxlen=10))
        logged_encodings = []
        SIMILARITY_THRESHOLD = 0.6
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
                
            face_boxes = self.get_face_box(frame)
            
            for i, face_box in enumerate(face_boxes):
                face = frame[max(0, face_box[1]-20):min(face_box[3]+20, frame.shape[0]-1),
                            max(0, face_box[0]-20):min(face_box[2]+20, frame.shape[1]-1)]
                
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)
                
                is_new = True
                if encodings:
                    encoding = encodings[0]
                    for logged in logged_encodings:
                        dist = np.linalg.norm(encoding - logged)
                        if dist < SIMILARITY_THRESHOLD:
                            is_new = False
                            break
                    
                    if is_new:
                        result = self.predict_age_gender(face)
                        if result[0] is not None:
                            gender, age, gender_conf, age_conf = result
                            avg_confidence = (gender_conf + age_conf) / 2
                            
                            with open(log_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([i, gender, age, f"{avg_confidence:.3f}", 
                                               f'{face_box[0]},{face_box[1]},{face_box[2]},{face_box[3]}', 
                                               frame_count, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                            logged_encodings.append(encoding)
                
                # Smooth predictions using buffers
                bbox_key = (int((face_box[0]+face_box[2])/2), int((face_box[1]+face_box[3])/2))
                
                result = self.predict_age_gender(face)
                if result[0] is not None:
                    gender, age, gender_conf, age_conf = result
                    
                    # Add to buffers for smoothing
                    age_num = int(age.strip('()').split('-')[0]) if '-' in age else int(age.strip('()').split('-')[0])
                    age_buffers[bbox_key].append(age_num)
                    gender_buffers[bbox_key].append(gender)
                    
                    # Get smoothed predictions
                    if len(age_buffers[bbox_key]) > 0:
                        smoothed_age = int(round(np.mean(age_buffers[bbox_key])))
                        age_range = f"{smoothed_age-3}–{smoothed_age+3}"
                    else:
                        age_range = age
                    
                    # Get most common gender
                    if len(gender_buffers[bbox_key]) > 0:
                        from collections import Counter
                        smoothed_gender = Counter(gender_buffers[bbox_key]).most_common(1)[0][0]
                    else:
                        smoothed_gender = gender
                    
                    avg_confidence = (gender_conf + age_conf) / 2
                    label = f"{smoothed_gender} | Age: {age_range} | {avg_confidence:.2f}"
                else:
                    label = "Detection Failed"
                
                # Draw rectangle around face (blue)
                cv2.rectangle(frame, (face_box[0], face_box[1]), 
                             (face_box[2], face_box[3]), (60, 120, 255), 2)
                
                # Add labels
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                overlay = frame.copy()
                y1 = max(face_box[1] - label_size[1] - 16, 0)
                y2 = face_box[1]
                x1 = face_box[0]
                x2 = face_box[0] + label_size[0] + 16
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, label, (x1 + 8, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Improved Gender and Age Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'detected_frame_{frame_count}.jpg', frame)
                print(f"Frame saved as detected_frame_{frame_count}.jpg")
                frame_count += 1
                
        cap.release()
        cv2.destroyAllWindows()

def download_models():
    """Download required model files"""
    import urllib.request
    
    models = {
        'opencv_face_detector.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
        'opencv_face_detector_uint8.pb': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector_uint8.pb',
        'age_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_deploy.prototxt',
        'age_net.caffemodel': 'https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1',
        'gender_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_deploy.prototxt',
        'gender_net.caffemodel': 'https://www.dropbox.com/s/iyv83wz7hiqx7ag/gender_net.caffemodel?dl=1'
    }
    
    print("Downloading model files...")
    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists")

def main():
    parser = argparse.ArgumentParser(description='Improved Gender and Age Detection')
    parser.add_argument('--input', type=str, default='', help='Path to input image or video file (not needed for webcam mode)')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], 
                       default='webcam', help='Detection mode (default: webcam)')
    parser.add_argument('--download', action='store_true', 
                       help='Download required model files')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Minimum confidence threshold for face detection')
    
    args = parser.parse_args()
    
    if args.download:
        download_models()
        return
    
    # Check if model files exist
    required_files = [
        'opencv_face_detector.pbtxt', 'opencv_face_detector_uint8.pb',
        'age_deploy.prototxt', 'age_net.caffemodel',
        'gender_deploy.prototxt', 'gender_net.caffemodel'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Missing model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Run with --download flag to download them automatically")
        return
    
    # Initialize detector
    detector = ImprovedGenderAgeDetector()
    
    # Update confidence threshold if provided
    if args.confidence:
        detector.min_face_confidence = args.confidence
        detector.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=args.confidence
        )
    
    if args.mode == 'image':
        if not args.input:
            print("Please provide input image path with --input")
            return
        result = detector.process_image(args.input)
        if result is not None:
            cv2.imshow('Improved Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Save result
            output_path = f"improved_result_{os.path.basename(args.input)}"
            cv2.imwrite(output_path, result)
            print(f"Result saved as {output_path}")
    elif args.mode == 'video':
        # Check if video file exists
        if not args.input or not os.path.exists(args.input):
            print(f"Video file not found: {args.input}")
            print("Please provide a valid video file path with --input")
            return
        print(f"Processing video: {args.input}")
        detector.process_video(args.input)
    else:  # webcam
        print("Processing webcam input...")
        detector.process_video(0)

if __name__ == "__main__":
    main() 