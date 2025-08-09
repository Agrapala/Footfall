import cv2
import numpy as np
import face_recognition
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
import hashlib
import time
from unique_count_logger import log_unique_count

class AgeGenderDetector:
    """Age and Gender detection for verified human faces with advanced accuracy"""
    
    def __init__(self):
        # Define the model files
        self.age_proto = "age_deploy.prototxt"
        self.age_model = "age_net.caffemodel"
        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "gender_net.caffemodel"
        
        # Load networks
        self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)
        self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
        
        # More precise age ranges for better accuracy
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                        '(38-43)', '(48-53)', '(60-100)']
        
        # Enhanced age mapping with more precise ranges and confidence weights
        self.age_mapping = {
            0: {'label': '0-2 years (Infant/Toddler)', 'weight': 1.2, 'confidence_boost': 0.1},
            1: {'label': '4-6 years (Preschool)', 'weight': 1.1, 'confidence_boost': 0.05},
            2: {'label': '8-12 years (Child)', 'weight': 1.0, 'confidence_boost': 0.0},
            3: {'label': '15-20 years (Teen/Young Adult)', 'weight': 1.0, 'confidence_boost': 0.0},
            4: {'label': '25-32 years (Young Adult)', 'weight': 1.0, 'confidence_boost': 0.0},
            5: {'label': '38-43 years (Adult)', 'weight': 1.0, 'confidence_boost': 0.0},
            6: {'label': '48-53 years (Middle Age)', 'weight': 1.0, 'confidence_boost': 0.0},
            7: {'label': '60+ years (Senior)', 'weight': 1.1, 'confidence_boost': 0.05}
        }
        
        self.gender_list = ['Male', 'Female']
        
        # Model parameters
        self.model_mean = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Adaptive confidence thresholds based on face size and quality
        self.base_age_confidence = 0.7
        self.base_gender_confidence = 0.8
        
        # Face preprocessing parameters
        self.face_size = (227, 227)
        
        # Advanced prediction tracking
        self.age_predictions = defaultdict(lambda: deque(maxlen=15))  # Increased history
        self.gender_predictions = defaultdict(lambda: deque(maxlen=10))
        self.face_quality_scores = defaultdict(lambda: deque(maxlen=5))
        
        # Ensemble prediction settings
        self.use_ensemble = True
        self.ensemble_weight_recent = 0.6
        self.ensemble_weight_quality = 0.4
        
    def analyze_face_quality(self, face_img):
        """Analyze face quality for better prediction confidence"""
        if face_img is None or face_img.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Calculate face size quality (larger faces are better)
        face_area = face_img.shape[0] * face_img.shape[1]
        size_quality = min(face_area / 10000, 1.0)  # Normalize to 0-1
        
        # Calculate overall quality score
        quality_score = (
            min(sharpness / 1000, 1.0) * 0.4 +  # Sharpness (40%)
            min(brightness / 255, 1.0) * 0.2 +   # Brightness (20%)
            min(contrast / 100, 1.0) * 0.2 +     # Contrast (20%)
            size_quality * 0.2                     # Size (20%)
        )
        
        return quality_score
    
    def detect_facial_features(self, face_img):
        """Detect facial features to improve age/gender prediction (robust to missing nose cascade)"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Load facial feature cascades
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # Try to load the nose cascade from local directory or OpenCV data
        nose_cascade_path = 'haarcascade_mcs_nose.xml'
        if not os.path.exists(nose_cascade_path):
            # Try OpenCV's data folder (may not exist)
            nose_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_mcs_nose.xml')
        nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
        
        # Detect features
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
        noses = []
        if not nose_cascade.empty():
            noses = nose_cascade.detectMultiScale(gray, 1.1, 3)
        # If nose cascade is missing, skip nose detection
        
        # Calculate feature-based confidence boost
        feature_score = 0.0
        
        # Eye detection quality
        if len(eyes) >= 2:
            feature_score += 0.3
        elif len(eyes) == 1:
            feature_score += 0.15
        
        # Nose detection quality
        if len(noses) >= 1:
            feature_score += 0.2
        
        # Face proportion analysis (child vs adult)
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            aspect_ratio = face_img.shape[1] / face_img.shape[0]
            # Children typically have different face proportions
            if 0.7 < aspect_ratio < 1.3:  # Normal face proportions
                feature_score += 0.2
            elif 0.5 < aspect_ratio < 1.5:  # Acceptable proportions
                feature_score += 0.1
        
        return min(feature_score, 1.0)
    
    def enhance_lighting_for_prediction(self, face_img):
        """Enhanced lighting compensation with multiple methods"""
        if face_img is None or face_img.size == 0:
            return face_img
            
        # Method 1: LAB color space enhancement
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Method 2: Analyze lighting conditions and apply appropriate correction
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
        
        # Method 3: Advanced contrast enhancement
        img_float = enhanced_bgr.astype(np.float32) / 255.0
        contrast_factor = 1.2
        brightness_factor = 0.1
        enhanced_float = img_float * contrast_factor + brightness_factor
        enhanced_float = np.clip(enhanced_float, 0, 1)
        enhanced_bgr = (enhanced_float * 255).astype(np.uint8)
        
        # Method 4: Histogram equalization for better feature extraction
        yuv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        enhanced_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return enhanced_bgr
    
    def preprocess_face(self, face_img):
        """Advanced preprocessing with multiple enhancement methods"""
        if face_img is None or face_img.size == 0:
            return None
            
        # Resize to standard size
        face_img = cv2.resize(face_img, self.face_size)
        
        # Apply enhanced lighting compensation
        face_img = self.enhance_lighting_for_prediction(face_img)
        
        # Apply histogram equalization for better contrast
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply Gaussian blur to reduce noise
        face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
        
        # Additional sharpening for better feature extraction
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        face_img = cv2.filter2D(face_img, -1, kernel)
        
        return face_img
    
    def get_adaptive_confidence_thresholds(self, face_img, face_quality):
        """Get adaptive confidence thresholds based on face quality"""
        # Base thresholds
        age_threshold = self.base_age_confidence
        gender_threshold = self.base_gender_confidence
        
        # Adjust based on face quality
        if face_quality > 0.8:  # High quality face
            age_threshold -= 0.1
            gender_threshold -= 0.1
        elif face_quality < 0.4:  # Low quality face
            age_threshold += 0.1
            gender_threshold += 0.1
        
        # Adjust based on face size
        face_area = face_img.shape[0] * face_img.shape[1]
        if face_area > 15000:  # Large face
            age_threshold -= 0.05
            gender_threshold -= 0.05
        elif face_area < 5000:  # Small face
            age_threshold += 0.05
            gender_threshold += 0.05
        
        return max(age_threshold, 0.5), max(gender_threshold, 0.6)
    
    def get_ensemble_prediction(self, face_id, current_prediction, confidence):
        """Get ensemble prediction using temporal smoothing"""
        if not self.use_ensemble or face_id not in self.age_predictions:
            return current_prediction, confidence
        
        # Add current prediction to history
        self.age_predictions[face_id].append((current_prediction, confidence))
        
        # Get recent predictions (last 5)
        recent_predictions = list(self.age_predictions[face_id])[-5:]
        
        if len(recent_predictions) < 3:
            return current_prediction, confidence
        
        # Calculate weighted ensemble
        total_weight = 0
        weighted_prediction = 0
        
        for i, (pred, conf) in enumerate(recent_predictions):
            weight = self.ensemble_weight_recent ** (len(recent_predictions) - i - 1)
            total_weight += weight
            weighted_prediction += pred * weight * conf
        
        if total_weight > 0:
            ensemble_prediction = weighted_prediction / total_weight
            ensemble_confidence = np.mean([conf for _, conf in recent_predictions])
            return ensemble_prediction, ensemble_confidence
        
        return current_prediction, confidence
    
    def get_more_precise_age(self, age_predictions, confidence, face_img=None):
        """Get more precise age prediction with advanced analysis and better child detection"""
        age_index = np.argmax(age_predictions)
        max_confidence = np.max(age_predictions)
        
        # Analyze face quality if available
        face_quality = 0.5  # Default
        if face_img is not None:
            face_quality = self.analyze_face_quality(face_img)
        
        # Detect facial features for additional confidence
        feature_confidence = 0.0
        if face_img is not None:
            feature_confidence = self.detect_facial_features(face_img)
        
        # Enhanced child detection using facial proportions
        child_indicators = 0
        if face_img is not None:
            child_indicators = self.analyze_child_characteristics(face_img)
        
        # Adjust confidence based on quality, features, and child indicators
        adjusted_confidence = max_confidence + (face_quality * 0.2) + (feature_confidence * 0.1) + (child_indicators * 0.15)
        adjusted_confidence = min(adjusted_confidence, 1.0)
        
        # Enhanced age prediction logic
        if adjusted_confidence > 0.85:
            # High confidence - use exact prediction with child correction
            age_info = self.age_mapping[age_index]
            final_age = self.correct_age_prediction(age_index, age_predictions, face_img)
            return final_age, adjusted_confidence
        
        # For lower confidence, consider secondary predictions with child bias
        sorted_indices = np.argsort(age_predictions)[::-1]
        top_3_indices = sorted_indices[:3]
        top_3_confidences = age_predictions[top_3_indices]
        
        # Enhanced child detection logic
        if self.is_likely_child(face_img, age_predictions):
            # If face shows child characteristics, bias towards child age ranges
            child_age = self.get_child_age_prediction(age_predictions, face_img)
            return child_age, adjusted_confidence + 0.1
        
        # If top predictions are close, provide a range
        if abs(top_3_confidences[0] - top_3_confidences[1]) < 0.2:
            age1_info = self.age_mapping[top_3_indices[0]]
            age2_info = self.age_mapping[top_3_indices[1]]
            return f"{age1_info['label']} or {age2_info['label']}", adjusted_confidence
        
        # Apply age-specific confidence adjustments and corrections
        age_info = self.age_mapping[age_index]
        final_confidence = adjusted_confidence + age_info['confidence_boost']
        final_confidence = min(final_confidence, 1.0)
        
        # Apply final age correction
        final_age = self.correct_age_prediction(age_index, age_predictions, face_img)
        
        return final_age, final_confidence
    
    def analyze_child_characteristics(self, face_img):
        """Analyze facial characteristics specific to children"""
        if face_img is None:
            return 0.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        indicators = 0.0
        
        # 1. Face proportion analysis (children have different proportions)
        height, width = face_img.shape[:2]
        aspect_ratio = width / height
        
        # Children typically have rounder faces
        if 0.8 < aspect_ratio < 1.2:
            indicators += 0.2
        
        # 2. Eye size relative to face (children have larger eyes)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(eyes) >= 2:
            # Calculate average eye size
            eye_areas = [w * h for (x, y, w, h) in eyes]
            avg_eye_area = np.mean(eye_areas)
            face_area = height * width
            eye_to_face_ratio = avg_eye_area / face_area
            
            # Children have larger eyes relative to face
            if eye_to_face_ratio > 0.01:  # Threshold for child-like proportions
                indicators += 0.3
        
        # 3. Skin texture analysis (children have smoother skin)
        # Calculate texture variance - children have less texture
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        if texture_variance < 500:  # Lower texture indicates younger age
            indicators += 0.2
        
        # 4. Face size analysis (children's faces are typically smaller in detection)
        face_area = height * width
        if face_area < 8000:  # Smaller detected faces often indicate children
            indicators += 0.1
        
        return min(indicators, 1.0)
    
    def is_likely_child(self, face_img, age_predictions):
        """Determine if the face is likely a child based on multiple factors"""
        if face_img is None:
            return False
        
        # 1. Check if any child age predictions are high
        child_indices = [0, 1, 2]  # 0-2, 4-6, 8-12 years
        child_confidence = max([age_predictions[i] for i in child_indices])
        
        # 2. Analyze facial characteristics
        child_indicators = self.analyze_child_characteristics(face_img)
        
        # 3. Check face proportions
        height, width = face_img.shape[:2]
        aspect_ratio = width / height
        
        # Combine all factors
        is_child = (
            child_confidence > 0.3 or  # High child age prediction
            child_indicators > 0.4 or  # Strong child characteristics
            (0.7 < aspect_ratio < 1.3 and child_confidence > 0.2)  # Good proportions + some child prediction
        )
        
        return is_child
    
    def get_child_age_prediction(self, age_predictions, face_img):
        """Get more accurate child age prediction"""
        child_indices = [0, 1, 2]  # 0-2, 4-6, 8-12 years
        child_confidences = [age_predictions[i] for i in child_indices]
        
        # Find the highest child age prediction
        best_child_index = child_indices[np.argmax(child_confidences)]
        best_child_confidence = max(child_confidences)
        
        # Analyze facial characteristics for more precise child age
        child_indicators = self.analyze_child_characteristics(face_img)
        
        # Refine child age based on characteristics
        if best_child_index == 0:  # 0-2 years
            if child_indicators > 0.6:
                return "0-2 years (Infant/Toddler)"
            else:
                return "0-2 years (Baby)"
        elif best_child_index == 1:  # 4-6 years
            if child_indicators > 0.5:
                return "4-6 years (Preschool)"
            else:
                return "4-6 years (Young Child)"
        else:  # 8-12 years
            if child_indicators > 0.4:
                return "8-12 years (Child)"
            else:
                return "8-12 years (Pre-teen)"
    
    def correct_age_prediction(self, age_index, age_predictions, face_img):
        """Apply corrections to age predictions based on facial analysis"""
        age_info = self.age_mapping[age_index]
        base_label = age_info['label']
        
        # Check if this might be a child misclassified as adult
        if age_index >= 3 and face_img is not None:  # Adult age ranges
            child_indicators = self.analyze_child_characteristics(face_img)
            child_confidence = max([age_predictions[i] for i in [0, 1, 2]])
            
            # If strong child indicators and some child prediction, correct to child
            if child_indicators > 0.5 and child_confidence > 0.2:
                return self.get_child_age_prediction(age_predictions, face_img)
        
        # Check if this might be an adult misclassified as child
        if age_index <= 2 and face_img is not None:  # Child age ranges
            child_indicators = self.analyze_child_characteristics(face_img)
            
            # If weak child indicators, might be young adult
            if child_indicators < 0.3:
                adult_confidence = max([age_predictions[i] for i in [3, 4, 5]])
                if adult_confidence > 0.3:
                    return "15-20 years (Teen/Young Adult)"
        
        # Apply confidence-based corrections
        max_confidence = np.max(age_predictions)
        if max_confidence < 0.6:
            # Low confidence - provide range
            sorted_indices = np.argsort(age_predictions)[::-1]
            second_best = sorted_indices[1]
            second_confidence = age_predictions[second_best]
            
            if abs(max_confidence - second_confidence) < 0.15:
                second_label = self.age_mapping[second_best]['label']
                return f"{base_label} or {second_label}"
        
        return base_label
    
    def predict_age_gender(self, face, face_id=None):
        """Advanced age and gender prediction with ensemble methods"""
        # Analyze face quality
        face_quality = self.analyze_face_quality(face)
        
        # Get adaptive confidence thresholds
        age_threshold, gender_threshold = self.get_adaptive_confidence_thresholds(face, face_quality)
        
        # Try multiple lighting enhancement methods for better accuracy
        enhanced_versions = []
        
        # Version 1: Standard enhancement
        enhanced_versions.append(self.enhance_lighting_for_prediction(face))
        
        # Version 2: Additional contrast enhancement
        enhanced_face2 = self.enhance_lighting_for_prediction(face)
        lab = cv2.cvtColor(enhanced_face2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced_versions.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        
        # Version 3: Normalized lighting
        enhanced_face3 = face.copy()
        img_float = enhanced_face3.astype(np.float32) / 255.0
        mean_val = np.mean(img_float)
        std_val = np.std(img_float)
        if std_val > 0:
            normalized = (img_float - mean_val) / std_val * 0.2 + 0.5
            normalized = np.clip(normalized, 0, 1)
            enhanced_versions.append((normalized * 255).astype(np.uint8))
        else:
            enhanced_versions.append(enhanced_face3)
        
        # Version 4: Enhanced for child detection
        enhanced_face4 = face.copy()
        enhanced_face4 = cv2.resize(enhanced_face4, (256, 256))
        enhanced_face4 = cv2.GaussianBlur(enhanced_face4, (3, 3), 0)
        enhanced_versions.append(enhanced_face4)
        
        # Version 5: Sharpened for better feature extraction
        enhanced_face5 = face.copy()
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced_face5 = cv2.filter2D(enhanced_face5, -1, kernel)
        enhanced_versions.append(enhanced_face5)
        
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
            
            if gender_confidence < gender_threshold:
                continue
                
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_confidence = age_preds[0].max()
            
            if age_confidence < age_threshold:
                continue
                
            # Get more precise age prediction with quality analysis
            precise_age, adjusted_age_confidence = self.get_more_precise_age(age_preds[0], age_confidence, enhanced_face)
            
            # Calculate combined confidence with quality boost
            combined_confidence = (gender_confidence + adjusted_age_confidence) / 2
            combined_confidence += face_quality * 0.1  # Quality boost
            
            # Keep the best result
            if combined_confidence > best_confidence:
                best_confidence = combined_confidence
                best_result = (gender, precise_age, gender_confidence, adjusted_age_confidence)
        
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
            
            if gender_confidence < gender_threshold:
                return None, None, 0, 0
                
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_confidence = age_preds[0].max()
            
            if age_confidence < age_threshold:
                return None, None, 0, 0
                
            # Get more precise age prediction
            precise_age, adjusted_age_confidence = self.get_more_precise_age(age_preds[0], age_confidence, face)
            
            return gender, precise_age, gender_confidence, adjusted_age_confidence
        
        return best_result

class LivenessDetector:
    """Detect if a face is from a real person or a photo"""
    
    def __init__(self):
        # Eye cascade for blink detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Extremely strict liveness detection parameters
        self.min_blinks = 6  # Much higher minimum blinks required
        self.movement_threshold = 20.0  # Extremely high movement threshold
        self.texture_threshold = 0.05  # Extremely strict texture threshold
        self.min_analysis_time = 15.0  # Much longer analysis time
        self.motion_threshold = 10.0  # Much higher motion threshold
        self.min_direction_changes = 40  # Much higher direction changes
        
        # Tracking for each face
        self.face_tracks = {}
        
        # Frame differencing for motion detection
        self.previous_frames = {}
        
        # Lighting compensation parameters
        self.lighting_adaptation = True
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
    def enhance_lighting(self, image):
        """Enhance image for better detection under different lighting conditions"""
        # Convert to LAB color space for better lighting compensation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        l_clahe = self.clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction for very dark or bright images
        gamma = 1.0
        mean_intensity = np.mean(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY))
        
        if mean_intensity < 50:  # Dark image
            gamma = 1.5
        elif mean_intensity > 200:  # Bright image
            gamma = 0.7
        
        if gamma != 1.0:
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_bgr = cv2.LUT(enhanced_bgr, table)
        
        return enhanced_bgr
        
    def detect_blinks(self, face_image, face_id):
        """Detect blinks with extremely strict criteria"""
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection
        gray_face = cv2.equalizeHist(gray_face)
        
        eyes = self.eye_cascade.detectMultiScale(
            gray_face, 
            scaleFactor=1.1, 
            minNeighbors=15,  # Much higher for more reliable detection
            minSize=(30, 30),
            maxSize=(60, 60)
        )
        
        if face_id not in self.face_tracks:
            self.face_tracks[face_id] = {
                'blink_count': 0,
                'last_eye_count': len(eyes),
                'last_blink_time': time.time(),
                'head_positions': [],
                'texture_scores': [],
                'frame_differences': [],
                'first_seen': time.time(),
                'eye_detection_history': [],
                'blink_patterns': []
            }
        
        track = self.face_tracks[face_id]
        current_eye_count = len(eyes)
        
        # Store eye detection history
        track['eye_detection_history'].append(current_eye_count)
        if len(track['eye_detection_history']) > 100:  # Much more history
            track['eye_detection_history'] = track['eye_detection_history'][-100:]
        
        # Extremely sophisticated blink detection
        if len(track['eye_detection_history']) > 20:
            recent_eyes = track['eye_detection_history'][-20:]
            
            # Detect blink with extremely strict criteria
            if (track['last_eye_count'] >= 2 and 
                current_eye_count < 2 and 
                time.time() - track['last_blink_time'] > 1.5):
                
                # Look for complex blink pattern (2 eyes -> 0-1 eyes -> 2 eyes -> 0-1 eyes -> 2 eyes)
                if len(recent_eyes) >= 10:
                    # Check for multiple blink patterns
                    pattern_count = 0
                    for i in range(len(recent_eyes) - 8):
                        if (recent_eyes[i] >= 2 and 
                            recent_eyes[i+1] < 2 and 
                            recent_eyes[i+2] < 2 and 
                            recent_eyes[i+3] >= 2 and
                            recent_eyes[i+4] < 2 and
                            recent_eyes[i+5] >= 2):
                            pattern_count += 1
                    
                    if pattern_count >= 2:  # Require multiple complex patterns
                        track['blink_count'] += 1
                        track['last_blink_time'] = time.time()
                        track['blink_patterns'].append(time.time())
        
        track['last_eye_count'] = current_eye_count
        return track['blink_count']
    
    def analyze_frame_differences(self, face_image, face_id):
        """Analyze frame differences with extremely strict motion detection"""
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        if face_id not in self.face_tracks:
            return False
        
        track = self.face_tracks[face_id]
        
        # Store current frame
        if face_id not in self.previous_frames:
            self.previous_frames[face_id] = gray_face
            return False
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray_face, self.previous_frames[face_id])
        motion_score = np.mean(frame_diff)
        
        track['frame_differences'].append(motion_score)
        if len(track['frame_differences']) > 50:  # Much more history
            track['frame_differences'] = track['frame_differences'][-50:]
        
        # Update previous frame
        self.previous_frames[face_id] = gray_face
        
        # Extremely strict motion analysis
        if len(track['frame_differences']) > 25:  # Much more samples required
            motion_variance = np.var(track['frame_differences'])
            mean_motion = np.mean(track['frame_differences'])
            
            # Require extremely high motion and variance
            return (mean_motion > self.motion_threshold and 
                   motion_variance > 8.0 and  # Much higher variance
                   motion_variance > mean_motion * 0.8)  # Much higher variance ratio
        
        return False
    
    def analyze_head_movement(self, face_bbox, face_id):
        """Analyze head movement patterns with extremely strict detection"""
        if face_id not in self.face_tracks:
            return False
        
        track = self.face_tracks[face_id]
        x, y, w, h = face_bbox
        center = (x + w/2, y + h/2)
        
        track['head_positions'].append(center)
        
        # Keep only recent positions
        if len(track['head_positions']) > 100:  # Much more history
            track['head_positions'] = track['head_positions'][-100:]
        
        # Calculate movement with extremely strict analysis
        if len(track['head_positions']) > 40:  # Much more positions required
            positions = np.array(track['head_positions'])
            
            # Calculate movement in both X and Y directions
            x_movement = np.std(positions[:, 0])
            y_movement = np.std(positions[:, 1])
            total_movement = np.sqrt(x_movement**2 + y_movement**2)
            
            # Also check for directional changes
            if len(positions) > 50:
                directions = np.diff(positions, axis=0)
                direction_changes = np.sum(np.abs(np.diff(directions, axis=0)))
                
                # Require both extremely high movement and direction changes
                return (total_movement > self.movement_threshold and 
                       direction_changes > self.min_direction_changes)
        
        return False
    
    def analyze_texture(self, face_image, face_id):
        """Enhanced texture analysis with extremely strict detection"""
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Multiple texture analysis methods
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        
        # Sobel gradient analysis
        grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_variance = gradient_magnitude.var()
        
        # Additional texture analysis
        gaussian_blur = cv2.GaussianBlur(gray_face, (5, 5), 0)
        blur_variance = gaussian_blur.var()
        
        # Combine texture metrics
        combined_texture_score = (laplacian_variance + gradient_variance + blur_variance) / 3
        
        if face_id not in self.face_tracks:
            return False
        
        track = self.face_tracks[face_id]
        track['texture_scores'].append(combined_texture_score)
        
        # Keep only recent scores
        if len(track['texture_scores']) > 40:  # Much more history
            track['texture_scores'] = track['texture_scores'][-40:]
        
        # Extremely strict texture analysis
        if len(track['texture_scores']) > 20:  # Much more samples required
            texture_variance = np.var(track['texture_scores'])
            mean_texture = np.mean(track['texture_scores'])
            
            # Photos have extremely low variance and consistent texture
            return (texture_variance < self.texture_threshold and 
                   mean_texture < 20)  # Much lower mean texture threshold
        
        return False
    
    def is_live_face(self, face_image, face_bbox, face_id, detection_time):
        """Determine if the face is from a real person with extremely strict criteria"""
        blink_count = self.detect_blinks(face_image, face_id)
        has_movement = self.analyze_head_movement(face_bbox, face_id)
        has_texture_variation = not self.analyze_texture(face_image, face_id)
        has_frame_motion = self.analyze_frame_differences(face_image, face_id)
        
        # Check if enough time has passed for analysis
        if face_id in self.face_tracks:
            time_elapsed = detection_time - self.face_tracks[face_id].get('first_seen', detection_time)
            
            # Require extremely long analysis time
            if time_elapsed < self.min_analysis_time:
                return True  # Assume live initially
            
            # Extremely strict criteria - require ALL indicators
            motion_indicators = 0
            if has_movement:
                motion_indicators += 1
            if has_frame_motion:
                motion_indicators += 1
            if blink_count >= self.min_blinks:
                motion_indicators += 1
            if has_texture_variation:
                motion_indicators += 1
            
            # Require ALL 4 indicators to be considered live
            is_live = motion_indicators == 4
            
            return is_live
        
        return True  # Assume live initially

class UniqueHumanCounter:
    def __init__(self, time_window_minutes=60, similarity_threshold=0.6):
        """
        Initialize the human counter
        
        Args:
            time_window_minutes: Time window to count unique humans
            similarity_threshold: Face similarity threshold (lower = more strict)
        """
        self.time_window = timedelta(minutes=time_window_minutes)
        self.similarity_threshold = similarity_threshold
        
        # Storage for known faces and their data
        self.known_faces = []  # Face encodings
        self.face_metadata = []  # Timestamps, IDs, etc.
        
        # Tracking data
        self.person_count = 0
        self.detection_history = deque(maxlen=1000)  # Recent detections
        self.time_period_counts = defaultdict(int)  # Counts per time period
        
        # Multiple face detection models for different angles
        self.face_cascades = {
            'frontal': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'profile': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml'),
            'alt': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            'alt2': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        }
        
        # Frame processing settings
        self.frame_skip = 3  # Process every 3rd frame for better detection
        self.frame_count = 0
        
        self.logged_person_ids = set()
        
        # Liveness detection
        self.liveness_detector = LivenessDetector()
        
        # Age and Gender detection
        self.age_gender_detector = AgeGenderDetector()
        
    def detect_faces_multi_angle(self, gray_image):
        """Detect faces using multiple cascade classifiers for different angles"""
        all_faces = []
        
        # Detect frontal faces with different classifiers
        frontal_faces = self.face_cascades['frontal'].detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        all_faces.extend(frontal_faces)
        
        # Detect profile faces
        profile_faces = self.face_cascades['profile'].detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        all_faces.extend(profile_faces)
        
        # Detect with alternative frontal classifiers
        alt_faces = self.face_cascades['alt'].detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        all_faces.extend(alt_faces)
        
        alt2_faces = self.face_cascades['alt2'].detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        all_faces.extend(alt2_faces)
        
        # Remove overlapping detections
        return self.remove_overlapping_faces(all_faces)
    
    def remove_overlapping_faces(self, faces):
        """Remove overlapping face detections"""
        if len(faces) == 0:
            return []
        
        # Convert to list of tuples (x, y, w, h)
        faces = [tuple(face) for face in faces]
        
        # Sort by area (largest first)
        faces.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        filtered_faces = []
        for face in faces:
            x, y, w, h = face
            overlap = False
            
            for existing_face in filtered_faces:
                ex, ey, ew, eh = existing_face
                
                # Calculate overlap
                x_overlap = max(0, min(x + w, ex + ew) - max(x, ex))
                y_overlap = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = x_overlap * y_overlap
                
                # If overlap is more than 50% of smaller face, skip
                smaller_area = min(w * h, ew * eh)
                if overlap_area > 0.5 * smaller_area:
                    overlap = True
                    break
            
            if not overlap:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def get_face_encoding(self, face_image):
        """Get face encoding from face image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            return encodings[0] if encodings else None
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None
    
    def find_matching_person(self, face_encoding):
        """Find if this face matches any known person"""
        if not self.known_faces or face_encoding is None:
            return -1
        
        # Compare with known faces
        face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] < self.similarity_threshold:
            return best_match_index
        else:
            return -1
    
    def add_new_person(self, face_encoding, face_image, timestamp):
        """Add a new person to the database"""
        person_id = len(self.known_faces)
        self.known_faces.append(face_encoding)
        
        metadata = {
            'person_id': person_id,
            'first_seen': timestamp,
            'last_seen': timestamp,
            'total_detections': 1,
            'face_image': face_image.copy()
        }
        self.face_metadata.append(metadata)
        
        # Log only if this person_id hasn't been logged yet
        if person_id not in self.logged_person_ids:
            log_unique_count("Face Recognition", person_id+1, self.time_window.total_seconds()//60, f"PersonID={person_id}")
            self.logged_person_ids.add(person_id)
        
        return person_id
    
    def update_person_data(self, person_id, timestamp):
        """Update existing person's data"""
        if 0 <= person_id < len(self.face_metadata):
            self.face_metadata[person_id]['last_seen'] = timestamp
            self.face_metadata[person_id]['total_detections'] += 1
    
    def cleanup_old_data(self, current_time):
        """Remove data outside the time window"""
        cutoff_time = current_time - self.time_window
        
        # Clean up detection history
        while (self.detection_history and 
               self.detection_history[0]['timestamp'] < cutoff_time):
            self.detection_history.popleft()
        
        # Update face metadata to mark inactive persons
        for metadata in self.face_metadata:
            if metadata['last_seen'] < cutoff_time:
                metadata['active'] = False
            else:
                metadata['active'] = True
    
    def get_unique_count_in_period(self, current_time):
        """Get count of unique humans in the current time window"""
        cutoff_time = current_time - self.time_window
        unique_persons = set()
        
        for detection in self.detection_history:
            if detection['timestamp'] >= cutoff_time:
                unique_persons.add(detection['person_id'])
        
        return len(unique_persons)
    
    def process_frame(self, frame):
        """Process a single frame and detect humans with age/gender detection"""
        self.frame_count += 1
        current_time = datetime.now()
        
        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            return frame, self.get_unique_count_in_period(current_time)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using multiple angles
        faces = self.detect_faces_multi_angle(gray)
        
        current_detections = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_image = frame[y:y+h, x:x+w]
            
            # Skip very small faces
            if w < 30 or h < 30:
                continue
            
            # Generate unique face ID for liveness tracking
            face_id = f"{x}_{y}_{w}_{h}"
            
            # Check liveness
            is_live = self.liveness_detector.is_live_face(face_image, (x, y, w, h), face_id, time.time())
            
            if not is_live:
                # Draw red box for detected photo/fake
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "FAKE", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue  # Skip processing fake faces
            
            # Get face encoding
            face_encoding = self.get_face_encoding(face_image)
            
            if face_encoding is not None:
                # Check if this is a known person
                person_id = self.find_matching_person(face_encoding)
                
                if person_id == -1:
                    # New person detected
                    person_id = self.add_new_person(face_encoding, face_image, current_time)
                    color = (0, 255, 0)  # Green for new person
                    label = f"New Person {person_id}"
                else:
                    # Known person
                    self.update_person_data(person_id, current_time)
                    color = (255, 0, 0)  # Blue for known person
                    label = f"Person {person_id}"
                
                # Record detection
                detection = {
                    'person_id': person_id,
                    'timestamp': current_time,
                    'bbox': (x, y, w, h)
                }
                self.detection_history.append(detection)
                current_detections.append(detection)
                
                # Predict age and gender for verified human faces
                age_gender_result = self.age_gender_detector.predict_age_gender(face_image)
                if age_gender_result[0] is not None:
                    gender, age, gender_conf, age_conf = age_gender_result
                    avg_confidence = (gender_conf + age_conf) / 2
                    age_gender_label = f"{gender} | {age} | {avg_confidence:.2f}"
                else:
                    age_gender_label = "Age/Gender: Unknown"
                
                # Draw bounding box and labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw person label
                cv2.putText(frame, label, (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw age/gender label
                cv2.putText(frame, age_gender_label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Cleanup old data
        self.cleanup_old_data(current_time)
        
        # Get current unique count
        unique_count = self.get_unique_count_in_period(current_time)
        
        return frame, unique_count
    
    def get_statistics(self):
        """Get detailed statistics"""
        current_time = datetime.now()
        cutoff_time = current_time - self.time_window
        
        # Active persons in current window
        active_persons = []
        for i, metadata in enumerate(self.face_metadata):
            if metadata['last_seen'] >= cutoff_time:
                active_persons.append({
                    'person_id': i,
                    'first_seen': metadata['first_seen'].strftime("%Y-%m-%d %H:%M:%S"),
                    'last_seen': metadata['last_seen'].strftime("%Y-%m-%d %H:%M:%S"),
                    'total_detections': metadata['total_detections']
                })
        
        return {
            'unique_count_in_window': len(active_persons),
            'total_persons_ever_seen': len(self.face_metadata),
            'time_window_minutes': self.time_window.total_seconds() / 60,
            'active_persons': active_persons,
            'current_time': current_time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_data(self, filename):
        """Save tracking data to file"""
        data = {
            'metadata': [],
            'settings': {
                'time_window_minutes': self.time_window.total_seconds() / 60,
                'similarity_threshold': self.similarity_threshold
            }
        }
        
        for i, metadata in enumerate(self.face_metadata):
            person_data = {
                'person_id': i,
                'first_seen': metadata['first_seen'].isoformat(),
                'last_seen': metadata['last_seen'].isoformat(),
                'total_detections': metadata['total_detections']
            }
            data['metadata'].append(person_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data saved to {filename}")


class SimplePersonCounter:
    """Simplified version using only body detection"""
    
    def __init__(self, time_window_minutes=60):
        self.time_window = timedelta(minutes=time_window_minutes)
        
        # Initialize HOG descriptor for person detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Tracking
        self.person_tracks = {}
        self.next_id = 0
        self.detection_history = deque(maxlen=500)
        
        # Basic liveness detection for body tracking
        self.body_tracks = {}
        self.previous_body_frames = {}
        
    def calculate_overlap(self, box1, box2):
        """Calculate overlap between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect_people_multi_scale(self, frame):
        """Detect people using multiple scales for better angle coverage"""
        all_detections = []
        
        # Original scale detection
        people, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(16, 16), scale=1.05, hitThreshold=0
        )
        all_detections.extend(people)
        
        # Smaller scale for distant people
        people_small, weights_small = self.hog.detectMultiScale(
            frame, winStride=(4, 4), padding=(8, 8), scale=1.02, hitThreshold=0
        )
        all_detections.extend(people_small)
        
        # Larger scale for close people
        people_large, weights_large = self.hog.detectMultiScale(
            frame, winStride=(16, 16), padding=(32, 32), scale=1.1, hitThreshold=0
        )
        all_detections.extend(people_large)
        
        # Remove overlapping detections
        return self.remove_overlapping_detections(all_detections)
    
    def remove_overlapping_detections(self, detections):
        """Remove overlapping person detections"""
        if len(detections) == 0:
            return []
        
        # Convert to list of tuples (x, y, w, h)
        detections = [tuple(det) for det in detections]
        
        # Sort by area (largest first)
        detections.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        filtered_detections = []
        for detection in detections:
            x, y, w, h = detection
            overlap = False
            
            for existing_detection in filtered_detections:
                ex, ey, ew, eh = existing_detection
                
                # Calculate overlap
                x_overlap = max(0, min(x + w, ex + ew) - max(x, ex))
                y_overlap = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = x_overlap * y_overlap
                
                # If overlap is more than 40% of smaller detection, skip
                smaller_area = min(w * h, ew * eh)
                if overlap_area > 0.4 * smaller_area:
                    overlap = True
                    break
            
            if not overlap:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def analyze_body_motion(self, frame, bbox, body_id):
        """Analyze body motion using frame differencing with extremely strict detection"""
        x, y, w, h = bbox
        
        # Extract body region
        body_region = frame[y:y+h, x:x+w]
        if body_region.size == 0:
            return False
        
        gray_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2GRAY)
        
        if body_id not in self.body_tracks:
            return False
        
        track = self.body_tracks[body_id]
        
        # Store current frame
        if body_id not in self.previous_body_frames:
            self.previous_body_frames[body_id] = gray_body
            return False
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray_body, self.previous_body_frames[body_id])
        motion_score = np.mean(frame_diff)
        
        track['motion_scores'].append(motion_score)
        if len(track['motion_scores']) > 50:  # Much more history
            track['motion_scores'] = track['motion_scores'][-50:]
        
        # Update previous frame
        self.previous_body_frames[body_id] = gray_body
        
        # Extremely strict motion analysis
        if len(track['motion_scores']) > 25:  # Much more samples required
            motion_variance = np.var(track['motion_scores'])
            mean_motion = np.mean(track['motion_scores'])
            
            # Require extremely high motion and variance
            return (mean_motion > 8.0 and  # Much higher motion threshold
                   motion_variance > 6.0 and  # Much higher variance threshold
                   motion_variance > mean_motion * 0.6)  # Much higher variance ratio
        
        return False
    
    def check_body_liveness(self, frame, bbox, detection_time):
        """Check if body detection is from a real person with extremely strict criteria"""
        x, y, w, h = bbox
        body_id = f"{x}_{y}_{w}_{h}"
        
        if body_id not in self.body_tracks:
            self.body_tracks[body_id] = {
                'positions': [],
                'first_seen': detection_time,
                'movement_detected': False,
                'motion_scores': [],
                'direction_changes': 0,
                'velocity_history': []
            }
        
        track = self.body_tracks[body_id]
        center = (x + w/2, y + h/2)
        track['positions'].append(center)
        
        # Keep only recent positions
        if len(track['positions']) > 80:  # Much more history
            track['positions'] = track['positions'][-80:]
        
        # Calculate velocity and movement patterns
        if len(track['positions']) > 5:
            positions = np.array(track['positions'])
            velocities = np.diff(positions, axis=0)
            track['velocity_history'].extend(velocities.tolist())
            
            # Keep only recent velocities
            if len(track['velocity_history']) > 40:
                track['velocity_history'] = track['velocity_history'][-40:]
        
        # Check for movement with extremely strict criteria
        has_position_movement = False
        if len(track['positions']) > 30:  # Much more positions required
            positions = np.array(track['positions'])
            
            # Calculate movement in both X and Y directions
            x_movement = np.std(positions[:, 0])
            y_movement = np.std(positions[:, 1])
            total_movement = np.sqrt(x_movement**2 + y_movement**2)
            
            # Calculate direction changes
            if len(positions) > 40:
                directions = np.diff(positions, axis=0)
                direction_changes = np.sum(np.abs(np.diff(directions, axis=0)))
                track['direction_changes'] = direction_changes
            
            # Calculate velocity variance
            has_velocity_variation = False
            if len(track['velocity_history']) > 15:
                velocities = np.array(track['velocity_history'])
                velocity_variance = np.var(velocities)
                has_velocity_variation = velocity_variance > 10.0  # Much higher threshold
            
            # Extremely strict movement requirements
            if (total_movement > 15.0 and  # Much higher movement threshold
                track['direction_changes'] > 50 and  # Much more direction changes required
                has_velocity_variation):
                has_position_movement = True
        
        # Check for frame motion
        has_frame_motion = self.analyze_body_motion(frame, bbox, body_id)
        
        # Require extremely long analysis time
        time_elapsed = detection_time - track['first_seen']
        if time_elapsed < 12.0:  # Much longer analysis time
            return True  # Assume live initially
        
        # Require both position movement and frame motion
        return has_position_movement and has_frame_motion
    
    def process_frame(self, frame):
        """Process frame for person detection"""
        current_time = datetime.now()
        
        # Detect people using multiple scales
        people = self.detect_people_multi_scale(frame)
        
        current_detections = []
        
        for (x, y, w, h) in people:
            # Check liveness for body detection
            is_live = self.check_body_liveness(frame, (x, y, w, h), time.time())
            
            if not is_live:
                # Draw red box for detected static image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "STATIC", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue  # Skip processing static detections
            
            # Simple tracking by overlap with previous detections
            matched = False
            best_overlap = 0
            best_id = None
            
            for track_id, track_data in self.person_tracks.items():
                if current_time - track_data['last_seen'] < timedelta(seconds=3):  # Reduced time window
                    overlap = self.calculate_overlap((x, y, w, h), track_data['last_bbox'])
                    if overlap > 0.2 and overlap > best_overlap:  # Lower threshold for better tracking
                        best_overlap = overlap
                        best_id = track_id
                        matched = True
            
            if matched:
                person_id = best_id
                self.person_tracks[person_id]['last_seen'] = current_time
                self.person_tracks[person_id]['last_bbox'] = (x, y, w, h)
            else:
                person_id = self.next_id
                self.next_id += 1
                self.person_tracks[person_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_bbox': (x, y, w, h)
                }
            
            # Record detection
            detection = {
                'person_id': person_id,
                'timestamp': current_time,
                'bbox': (x, y, w, h)
            }
            self.detection_history.append(detection)
            current_detections.append(detection)
            
            # Draw detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {person_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Clean up old tracks
        cutoff_time = current_time - self.time_window
        active_persons = set()
        
        for detection in self.detection_history:
            if detection['timestamp'] >= cutoff_time:
                active_persons.add(detection['person_id'])
        
        return frame, len(active_persons)


def main():
    """Main function to run the human counter"""
    print("Choose counting method:")
    print("1. Face Recognition (more accurate, requires face_recognition library)")
    print("2. Body Detection (faster, less accurate)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        try:
            counter = UniqueHumanCounter(time_window_minutes=30)
            method_name = "Face Recognition"
        except ImportError:
            print("face_recognition library not found. Using body detection instead.")
            counter = SimplePersonCounter(time_window_minutes=30)
            method_name = "Body Detection"
    else:
        counter = SimplePersonCounter(time_window_minutes=30)
        method_name = "Body Detection"
    
    # Initialize video capture with 1.mp4
    video_path = '2.mp4'
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please make sure 1.mp4 exists in the current directory")
        return
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Starting {method_name} Human Counter with video: {video_path}")
    print("Press 'q' to quit, 's' for statistics, 'r' to reset count")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        # Process frame
        processed_frame, unique_count = counter.process_frame(frame)
        
        # Display information
        info_text = f"Method: {method_name}"
        cv2.putText(processed_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        count_text = f"Unique Humans (30min): {unique_count}"
        cv2.putText(processed_frame, count_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        time_text = f"Time: {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(processed_frame, time_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Unique Human Counter', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if hasattr(counter, 'get_statistics'):
                stats = counter.get_statistics()
                print(f"\n=== Statistics ===")
                print(f"Unique count in window: {stats['unique_count_in_window']}")
                print(f"Total persons ever seen: {stats['total_persons_ever_seen']}")
                print(f"Time window: {stats['time_window_minutes']} minutes")
                print(f"Current time: {stats['current_time']}")
            else:
                print(f"\nCurrent unique count: {unique_count}")
        elif key == ord('r'):
            # Reset counter
            if hasattr(counter, 'known_faces'):
                counter.known_faces = []
                counter.face_metadata = []
                counter.detection_history.clear()
                counter.logged_person_ids.clear() # Clear logged IDs on reset
            else:
                counter.person_tracks = {}
                counter.detection_history.clear()
                counter.next_id = 0
            print("Counter reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save final statistics
    if hasattr(counter, 'save_data'):
        counter.save_data('human_count_data.json')
    
    print("Human counting session ended.")


if __name__ == "__main__":
    main()