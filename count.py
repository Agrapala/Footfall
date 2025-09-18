import cv2
import numpy as np
import face_recognition
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os
import hashlib
import time
import csv
try:
    from unique_count_logger import log_unique_count
except ImportError:
    def log_unique_count(*args, **kwargs):
        return None
from age_gender import ImprovedGenderAgeDetector

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
        self.enable_lighting_enhancement = False  # Disable by default for better webcam brightness
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
    def enhance_lighting(self, image):
        """Enhanced lighting compensation for outdoor conditions with multiple techniques"""
        # Return original image if enhancement is disabled
        if not self.enable_lighting_enhancement:
            return image
            
        # Convert to LAB color space for better lighting compensation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate lighting statistics
        mean_intensity = np.mean(l)
        std_intensity = np.std(l)
        
        # Dynamic CLAHE parameters based on lighting conditions
        if mean_intensity < 30:  # Very dark (night/indoor)
            clip_limit = 4.0
            tile_size = (4, 4)
        elif mean_intensity < 80:  # Dark (dawn/dusk)
            clip_limit = 3.5
            tile_size = (6, 6)
        elif mean_intensity > 200:  # Very bright (direct sunlight)
            clip_limit = 2.0
            tile_size = (12, 12)
        elif mean_intensity > 150:  # Bright (overcast)
            clip_limit = 2.5
            tile_size = (10, 10)
        else:  # Normal lighting
            clip_limit = 3.0
            tile_size = (8, 8)
        
        # Create dynamic CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)
        
        # Shadow and highlight correction
        l_corrected = self.shadow_highlight_correction(l_clahe, mean_intensity)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_corrected, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction for very dark or bright images
        gamma = self.calculate_dynamic_gamma(mean_intensity, std_intensity)
        
        if gamma != 1.0:
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_bgr = cv2.LUT(enhanced_bgr, table)
        
        # Final contrast enhancement
        enhanced_bgr = self.enhance_contrast(enhanced_bgr, mean_intensity)
        
        return enhanced_bgr
    
    def shadow_highlight_correction(self, l_channel, mean_intensity):
        """Correct shadows and highlights for better face detection"""
        # Create shadow and highlight masks
        shadow_mask = l_channel < mean_intensity * 0.5
        highlight_mask = l_channel > mean_intensity * 1.5
        
        # Apply different corrections
        corrected = l_channel.copy()
        
        # Brighten shadows
        if np.any(shadow_mask):
            shadow_boost = 1.3 if mean_intensity < 100 else 1.1
            corrected[shadow_mask] = np.clip(corrected[shadow_mask] * shadow_boost, 0, 255)
        
        # Darken highlights
        if np.any(highlight_mask):
            highlight_reduce = 0.8 if mean_intensity > 150 else 0.9
            corrected[highlight_mask] = np.clip(corrected[highlight_mask] * highlight_reduce, 0, 255)
        
        return corrected.astype(np.uint8)
    
    def calculate_dynamic_gamma(self, mean_intensity, std_intensity):
        """Calculate dynamic gamma based on lighting statistics"""
        # Base gamma calculation
        if mean_intensity < 40:  # Very dark
            gamma = 1.6
        elif mean_intensity < 80:  # Dark
            gamma = 1.3
        elif mean_intensity > 200:  # Very bright
            gamma = 0.6
        elif mean_intensity > 150:  # Bright
            gamma = 0.8
        else:  # Normal
            gamma = 1.0
        
        # Adjust based on contrast (std_intensity)
        if std_intensity < 20:  # Low contrast
            gamma *= 1.2
        elif std_intensity > 60:  # High contrast
            gamma *= 0.9
        
        return np.clip(gamma, 0.5, 2.0)
    
    def enhance_contrast(self, image, mean_intensity):
        """Enhance contrast based on lighting conditions"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast enhancement factor
        if mean_intensity < 50:  # Dark conditions
            alpha = 1.3  # Contrast enhancement
            beta = 10    # Brightness boost
        elif mean_intensity > 180:  # Bright conditions
            alpha = 0.8  # Reduce contrast
            beta = -20   # Reduce brightness
        else:  # Normal conditions
            alpha = 1.1
            beta = 0
        
        # Apply contrast enhancement
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Additional sharpening for better edge detection
        if mean_intensity > 100:  # Only for bright conditions
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced
        
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
        self.age_gender_detector = ImprovedGenderAgeDetector()
        
        # Data storage
        self.data_file = 'human_detection_data.csv'
        self.initialize_data_file()
    
    def initialize_data_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['datetime', 'person_id', 'gender', 'age', 'confidence', 'bbox', 'first_detection'])
    
    def save_detection_data(self, person_id, gender, age, confidence, bbox, is_first_detection=False):
        """Save detection data to CSV file"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, person_id, gender, age, f"{confidence:.3f}", bbox_str, is_first_detection])
        
    def detect_faces_multi_angle(self, gray_image):
        """Detect faces using multiple cascade classifiers with adaptive parameters for outdoor lighting"""
        all_faces = []
        
        # Analyze lighting conditions to adjust detection parameters
        mean_intensity = np.mean(gray_image)
        
        # Adaptive parameters based on lighting conditions
        if mean_intensity < 50:  # Dark conditions
            scale_factor = 1.05
            min_neighbors = 2
            min_size = (25, 25)
        elif mean_intensity > 180:  # Very bright conditions
            scale_factor = 1.15
            min_neighbors = 5
            min_size = (35, 35)
        elif mean_intensity > 120:  # Bright conditions
            scale_factor = 1.1
            min_neighbors = 4
            min_size = (30, 30)
        else:  # Normal conditions
            scale_factor = 1.1
            min_neighbors = 3
            min_size = (30, 30)
        
        # Detect frontal faces with adaptive parameters
        frontal_faces = self.face_cascades['frontal'].detectMultiScale(
            gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        all_faces.extend(frontal_faces)
        
        # Detect profile faces with slightly different parameters
        profile_faces = self.face_cascades['profile'].detectMultiScale(
            gray_image, scaleFactor=scale_factor, minNeighbors=max(2, min_neighbors-1), minSize=min_size
        )
        all_faces.extend(profile_faces)
        
        # Detect with alternative frontal classifiers
        alt_faces = self.face_cascades['alt'].detectMultiScale(
            gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        all_faces.extend(alt_faces)
        
        alt2_faces = self.face_cascades['alt2'].detectMultiScale(
            gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
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
    
    def add_new_person(self, face_encoding, face_image, timestamp, bbox=None):
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
        
        # Use original frame for display, enhanced frame only for detection
        enhanced_frame = self.liveness_detector.enhance_lighting(frame)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using multiple angles
        faces = self.detect_faces_multi_angle(gray)
        
        current_detections = []
        
        for (x, y, w, h) in faces:
            # Extract face region from enhanced frame
            face_image = enhanced_frame[y:y+h, x:x+w]
            
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
                    person_id = self.add_new_person(face_encoding, face_image, current_time, (x, y, w, h))
                    color = (0, 255, 0)  # Green for new person
                    label = f"New Person {person_id}"
                    is_new_person = True
                else:
                    # Known person
                    self.update_person_data(person_id, current_time)
                    color = (255, 0, 0)  # Blue for known person
                    label = f"Person {person_id}"
                    is_new_person = False
                
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
                    
                    # Save detection data to CSV only for new persons
                    if is_new_person:
                        self.save_detection_data(person_id, gender, age, avg_confidence, (x, y, w, h), True)
                else:
                    age_gender_label = "Age/Gender: Unknown"
                    # Save detection data even if age/gender prediction failed, only for new persons
                    if is_new_person:
                        self.save_detection_data(person_id, "Unknown", "Unknown", 0.0, (x, y, w, h), True)
                
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
        
        # Lighting enhancement for outdoor conditions
        self.enable_lighting_enhancement = False  # Disable by default for better webcam brightness
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    def enhance_frame_lighting(self, frame):
        """Enhance frame lighting for better outdoor person detection"""
        # Return original frame if enhancement is disabled
        if not self.enable_lighting_enhancement:
            return frame
            
        # Convert to LAB color space for better lighting compensation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate lighting statistics
        mean_intensity = np.mean(l)
        
        # Dynamic CLAHE parameters based on lighting conditions
        if mean_intensity < 40:  # Dark conditions
            clip_limit = 4.0
            tile_size = (6, 6)
        elif mean_intensity > 180:  # Very bright conditions
            clip_limit = 2.0
            tile_size = (12, 12)
        elif mean_intensity > 120:  # Bright conditions
            clip_limit = 2.5
            tile_size = (10, 10)
        else:  # Normal conditions
            clip_limit = 3.0
            tile_size = (8, 8)
        
        # Create dynamic CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction for very dark or bright images
        gamma = 1.0
        if mean_intensity < 50:  # Dark image
            gamma = 1.3
        elif mean_intensity > 200:  # Bright image
            gamma = 0.7
        
        if gamma != 1.0:
            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_bgr = cv2.LUT(enhanced_bgr, table)
        
        return enhanced_bgr
        
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
        """Process frame for person detection with outdoor lighting enhancement"""
        current_time = datetime.now()
        
        # Use enhanced frame for detection but original for display
        enhanced_frame = self.enhance_frame_lighting(frame)
        
        # Detect people using multiple scales
        people = self.detect_people_multi_scale(enhanced_frame)
        
        current_detections = []
        
        for (x, y, w, h) in people:
            # Check liveness for body detection using enhanced frame
            is_live = self.check_body_liveness(enhanced_frame, (x, y, w, h), time.time())
            
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
            
            # Draw detection on original frame for display
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
    try:
        counter = UniqueHumanCounter(time_window_minutes=30)
        method_name = "Face Recognition"
    except ImportError:
        print("face_recognition library not found. Using body detection instead.")
        counter = SimplePersonCounter(time_window_minutes=30)
        method_name = "Body Detection"
    
    # Initialize webcam capture with outdoor-optimized settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)
    
    # Optimize camera settings for balanced lighting
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Allow some auto exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # Less aggressive exposure
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 60)  # Slightly higher brightness
    cap.set(cv2.CAP_PROP_CONTRAST, 50)  # Moderate contrast
    cap.set(cv2.CAP_PROP_SATURATION, 50)  # Moderate saturation
    cap.set(cv2.CAP_PROP_GAIN, 0)  # Disable auto gain
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)  # Set white balance to daylight
    
    print(f"Starting {method_name} Human Counter with webcam")
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