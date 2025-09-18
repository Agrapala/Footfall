import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
import argparse
import json
import time
import csv
from collections import defaultdict, deque
try:
    from face_recognition_logger import log_face_recognition
except ImportError:
    def log_face_recognition(*args, **kwargs):
        return None

class ThiefDetectionSystem:
    """Enhanced thief detection system with outdoor lighting support"""
    
    def __init__(self, thief_faces_dir="thief_faces", thief_encodings_file="thief_encodings.pkl"):
        self.thief_faces_dir = thief_faces_dir
        self.thief_encodings_file = thief_encodings_file
        self.thief_encodings = []
        self.thief_names = []
        self.thief_metadata = []  # Store additional thief info
        self.detection_log = []
        self.alert_history = deque(maxlen=1000)
        
        # Create directories if they don't exist
        if not os.path.exists(thief_faces_dir):
            os.makedirs(thief_faces_dir)
        
        # Initialize CSV logging
        self.thief_log_file = 'thief_detections.csv'
        self.initialize_thief_log()
        
        # Load existing thief encodings
        self.load_thief_encodings()
        
        # Lighting enhancement for outdoor conditions
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    def initialize_thief_log(self):
        """Initialize CSV file for thief detection logging"""
        if not os.path.exists(self.thief_log_file):
            with open(self.thief_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'thief_name', 'confidence', 'location', 'camera_id', 'alert_level'])
    
    def log_thief_detection(self, thief_name, confidence, location, camera_id="Camera_0", alert_level="HIGH"):
        """Log thief detection to CSV file"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}" if location else "Unknown"
        
        with open(self.thief_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, thief_name, f"{confidence:.3f}", location_str, camera_id, alert_level])
    
    def enhance_lighting(self, image):
        """Enhanced lighting compensation for outdoor thief detection"""
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
        
        # Additional gamma correction
        gamma = self.calculate_dynamic_gamma(mean_intensity, std_intensity)
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_bgr = cv2.LUT(enhanced_bgr, table)
        
        return enhanced_bgr
    
    def shadow_highlight_correction(self, l_channel, mean_intensity):
        """Correct shadows and highlights for better thief detection"""
        shadow_mask = l_channel < mean_intensity * 0.5
        highlight_mask = l_channel > mean_intensity * 1.5
        
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
        
        # Adjust based on contrast
        if std_intensity < 20:  # Low contrast
            gamma *= 1.2
        elif std_intensity > 60:  # High contrast
            gamma *= 0.9
        
        return np.clip(gamma, 0.5, 2.0)
    
    def add_thief(self, image_path, thief_name, description="", threat_level="HIGH"):
        """Add a new thief to the database"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                print(f"❌ No face found in {image_path}")
                return False
            
            if len(encodings) > 1:
                print(f"⚠️  Multiple faces found in {image_path}. Using the first one.")
            
            # Add encoding and metadata
            self.thief_encodings.append(encodings[0])
            self.thief_names.append(thief_name)
            
            thief_info = {
                'name': thief_name,
                'description': description,
                'threat_level': threat_level,
                'date_added': datetime.now().isoformat(),
                'image_path': image_path
            }
            self.thief_metadata.append(thief_info)
            
            print(f"✅ Added THIEF: {thief_name} (Threat Level: {threat_level})")
            self.save_thief_encodings()
            
            # Log the addition
            log_face_recognition('thief_added', thief_name, '', 'thief', image_path, "Thief_System")
            return True
            
        except Exception as e:
            print(f"❌ Error adding thief {thief_name}: {str(e)}")
            return False
    
    def load_thief_encodings(self):
        """Load thief encodings from file"""
        if os.path.exists(self.thief_encodings_file):
            try:
                with open(self.thief_encodings_file, 'rb') as f:
                    data = pickle.load(f)
                self.thief_encodings = data['encodings']
                self.thief_names = data['names']
                self.thief_metadata = data.get('metadata', [])
                print(f"🔍 Loaded {len(self.thief_names)} known thieves")
            except Exception as e:
                print(f"❌ Error loading thief encodings: {str(e)}")
                self.thief_encodings = []
                self.thief_names = []
                self.thief_metadata = []
    
    def save_thief_encodings(self):
        """Save thief encodings to file"""
        data = {
            'encodings': self.thief_encodings,
            'names': self.thief_names,
            'metadata': self.thief_metadata
        }
        with open(self.thief_encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Thief encodings saved to {self.thief_encodings_file}")
    
    def detect_thieves_in_image(self, image_path, save_result=True):
        """Detect thieves in a single image"""
        try:
            # Load and enhance image
            image = face_recognition.load_image_file(image_path)
            enhanced_image = self.enhance_lighting(image)
            rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            detected_thieves = []
            alert_triggered = False
            
            # Process each face found
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.thief_encodings, face_encoding)
                name = "Unknown"
                confidence = 0
                threat_level = "UNKNOWN"
                
                # Calculate distances to known thieves
                if len(self.thief_encodings) > 0:
                    face_distances = face_recognition.face_distance(self.thief_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        name = self.thief_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        threat_level = self.thief_metadata[best_match_index].get('threat_level', 'HIGH')
                        alert_triggered = True
                        
                        # Log thief detection
                        self.log_thief_detection(name, confidence, (top, right, bottom, left))
                        print(f"🚨 ALERT: THIEF DETECTED - {name} (Confidence: {confidence:.2f}, Threat: {threat_level})")
                
                detected_thieves.append({
                    'name': name,
                    'confidence': confidence,
                    'threat_level': threat_level,
                    'location': (top, right, bottom, left),
                    'is_thief': name != "Unknown"
                })
                
                # Draw rectangle and label
                if save_result:
                    color = (0, 0, 255) if name != "Unknown" else (0, 255, 0)  # Red for thieves, green for others
                    thickness = 3 if name != "Unknown" else 2
                    
                    cv2.rectangle(rgb_image, (left, top), (right, bottom), color, thickness)
                    cv2.rectangle(rgb_image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    label = f"🚨 {name}" if name != "Unknown" else "Person"
                    cv2.putText(rgb_image, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                    
                    if name != "Unknown":
                        # Add threat level indicator
                        threat_text = f"THREAT: {threat_level}"
                        cv2.putText(rgb_image, threat_text, (left + 6, bottom - 20), font, 0.4, (0, 0, 255), 1)
            
            # Save result image if requested
            if save_result:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"thief_detection_{timestamp}.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                print(f"📸 Detection result saved to {output_path}")
            
            return detected_thieves, alert_triggered
            
        except Exception as e:
            print(f"❌ Error detecting thieves in {image_path}: {str(e)}")
            return [], False
    
    def start_thief_monitoring(self, camera_index=0, save_detections=True):
        """Start real-time thief monitoring with enhanced outdoor support"""
        cap = cv2.VideoCapture(camera_index)
        
        # Optimize camera settings for outdoor conditions
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
        cap.set(cv2.CAP_PROP_CONTRAST, 50)
        cap.set(cv2.CAP_PROP_SATURATION, 50)
        cap.set(cv2.CAP_PROP_GAIN, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🚨 Starting THIEF MONITORING System")
        print("Press 'q' to quit, 's' to save screenshot, 'a' to add current frame as thief")
        print("=" * 50)
        
        process_this_frame = True
        logged_thieves = set()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Apply lighting enhancement
            enhanced_frame = self.enhance_lighting(frame)
            
            # Resize frame for faster processing
            small_frame = cv2.resize(enhanced_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            if process_this_frame:
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                alert_detected = False
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.thief_encodings, face_encoding)
                    name = "Unknown"
                    confidence = 0
                    threat_level = "UNKNOWN"
                    
                    if len(self.thief_encodings) > 0:
                        face_distances = face_recognition.face_distance(self.thief_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                            name = self.thief_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            threat_level = self.thief_metadata[best_match_index].get('threat_level', 'HIGH')
                            alert_detected = True
                            
                            # Log detection
                            detection_info = {
                                'thief_name': name,
                                'timestamp': datetime.now().isoformat(),
                                'confidence': confidence,
                                'threat_level': threat_level,
                                'frame_number': frame_count
                            }
                            self.alert_history.append(detection_info)
                            
                            # Log to CSV
                            self.log_thief_detection(name, confidence, (0, 0, 0, 0), f"Camera_{camera_index}")
                            
                            print(f"🚨🚨🚨 THIEF ALERT: {name} detected! (Confidence: {confidence:.2f}, Threat: {threat_level}) 🚨🚨🚨")
                            
                            if name not in logged_thieves:
                                log_face_recognition('thief_detected', name, confidence, 'thief', '', f"Camera_{camera_index}")
                                logged_thieves.add(name)
                    
                    face_names.append(f"{name} ({confidence:.2f})")
                
                # Add alert overlay to frame
                if alert_detected:
                    cv2.putText(frame, "🚨 THIEF DETECTED! 🚨", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, "ALERT: SECURITY BREACH", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            process_this_frame = not process_this_frame
            
            # Display results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangle around face
                is_thief = "Unknown" not in name and name != "Unknown (0.00)"
                color = (0, 0, 255) if is_thief else (0, 255, 0)  # Red for thieves
                thickness = 4 if is_thief else 2
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                
                # Draw label with alert indicator
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                label = f"🚨 {name}" if is_thief else f"Person {name}"
                cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Add system status
            cv2.putText(frame, f"Thieves in DB: {len(self.thief_names)}", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('🚨 THIEF DETECTION SYSTEM 🚨', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'thief_screenshot_{timestamp}.jpg', frame)
                print(f"📸 Screenshot saved as thief_screenshot_{timestamp}.jpg")
            elif key == ord('a'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'thief_capture_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Frame captured as {filename}")
                print("Use this image to add a new thief to the database")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save detection log
        if self.alert_history and save_detections:
            with open(f'thief_alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(list(self.alert_history), f, indent=2)
            print(f"📊 Alert log saved with {len(self.alert_history)} detections")
    
    def list_thieves(self):
        """List all known thieves in the database"""
        if not self.thief_names:
            print("📋 No thieves in database")
            return
        
        print(f"📋 Known Thieves ({len(self.thief_names)}):")
        print("=" * 50)
        for i, (name, metadata) in enumerate(zip(self.thief_names, self.thief_metadata)):
            threat_level = metadata.get('threat_level', 'UNKNOWN')
            description = metadata.get('description', 'No description')
            date_added = metadata.get('date_added', 'Unknown')
            print(f"{i+1}. 🚨 {name}")
            print(f"   Threat Level: {threat_level}")
            print(f"   Description: {description}")
            print(f"   Added: {date_added}")
            print("-" * 30)
    
    def remove_thief(self, name):
        """Remove a thief from the database"""
        if name in self.thief_names:
            index = self.thief_names.index(name)
            del self.thief_names[index]
            del self.thief_encodings[index]
            del self.thief_metadata[index]
            self.save_thief_encodings()
            print(f"✅ Removed thief: {name}")
            log_face_recognition('thief_removed', name, '', 'thief', '', "Thief_System")
            return True
        else:
            print(f"❌ Thief '{name}' not found in database")
            return False
    
    def get_thief_statistics(self):
        """Get statistics about thief detections"""
        total_thieves = len(self.thief_names)
        threat_levels = defaultdict(int)
        
        for metadata in self.thief_metadata:
            threat_levels[metadata.get('threat_level', 'UNKNOWN')] += 1
        
        recent_alerts = len([alert for alert in self.alert_history 
                           if datetime.fromisoformat(alert['timestamp']) > 
                           datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)])
        
        return {
            'total_thieves': total_thieves,
            'threat_levels': dict(threat_levels),
            'recent_alerts_today': recent_alerts,
            'total_alerts': len(self.alert_history)
        }

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir="known_faces", encodings_file="face_encodings.pkl"):
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.detection_log = []
        
        # Create directories if they don't exist
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            
        # Load existing encodings
        self.load_encodings()
    
    def add_known_person(self, image_path, person_name):
        """Add a new person to the known faces database"""
        result = False
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                print(f"No face found in {image_path}")
                return False
            
            if len(encodings) > 1:
                print(f"Multiple faces found in {image_path}. Using the first one.")
            
            # Add encoding and name
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(person_name)
            
            print(f"Added {person_name} to known faces database")
            result = True
            # Log add event
            log_face_recognition('added', person_name, '', 'known', image_path, "Camera 0")
            return True
            
        except Exception as e:
            print(f"Error adding person {person_name}: {str(e)}")
            return False
    
    def load_faces_from_directory(self):
        """Load all faces from the known_faces directory"""
        if not os.path.exists(self.known_faces_dir):
            print(f"Directory {self.known_faces_dir} not found")
            return
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.known_faces_dir, filename)
                # Use filename (without extension) as person name
                person_name = os.path.splitext(filename)[0]
                self.add_known_person(image_path, person_name)
        
        # Save encodings after loading
        self.save_encodings()
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Encodings saved to {self.encodings_file}")
    
    def load_encodings(self):
        """Load face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} known faces")
            except Exception as e:
                print(f"Error loading encodings: {str(e)}")
                self.known_face_encodings = []
                self.known_face_names = []
    
    def detect_faces_in_image(self, image_path, save_result=False):
        """Detect and identify faces in a single image"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            detected_persons = []
            logged_names = set()
            
            # Process each face found
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 0
                
                # Calculate distances to known faces
                if len(self.known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                detected_persons.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left)
                })
                
                # Draw rectangle and label if saving result
                if save_result:
                    cv2.rectangle(rgb_image, (left, top), (right, bottom), (255, 0, 0) if name != "Unknown" else (0, 255, 0), 2)
                    cv2.rectangle(rgb_image, (left, bottom - 35), (right, bottom), (255, 0, 0) if name != "Unknown" else (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(rgb_image, f"{name} ({confidence:.2f})", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
                if name not in logged_names and name != "Unknown":
                    log_face_recognition('detected', name, confidence, 'known', image_path, "Camera 0")
                    logged_names.add(name)
                elif name not in logged_names:
                    log_face_recognition('detected', name, confidence, 'unknown', image_path, "Camera 0")
                    logged_names.add(name)
            
            # Save result image if requested
            if save_result:
                output_path = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                print(f"Result saved to {output_path}")
            
            return detected_persons
            
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {str(e)}")
            return []
    
    def start_video_monitoring(self, camera_index=0, save_detections=False):
        """Start real-time video monitoring"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting video monitoring. Press 'q' to quit, 's' to save screenshot")
        
        process_this_frame = True
        logged_names = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            if process_this_frame:
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = 0
                    
                    if len(self.known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                            name = self.known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            
                            # Log detection of known person
                            detection_info = {
                                'name': name,
                                'timestamp': datetime.now().isoformat(),
                                'confidence': confidence
                            }
                            self.detection_log.append(detection_info)
                            print(f"ALERT: {name} detected! (Confidence: {confidence:.2f})")
                            if name not in logged_names:
                                log_face_recognition('detected', name, confidence, 'known', '', "Camera 0")
                                logged_names.add(name)
                        else:
                            if name not in logged_names:
                                log_face_recognition('detected', name, confidence, 'unknown', '', "Camera 0")
                                logged_names.add(name)
                    
                    face_names.append(f"{name} ({confidence:.2f})")
            
            process_this_frame = not process_this_frame
            
            # Display results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangle around face
                color = (0, 0, 255) if "Unknown" not in name else (0, 255, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition Security System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'screenshot_{timestamp}.jpg', frame)
                print(f"Screenshot saved as screenshot_{timestamp}.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save detection log
        if self.detection_log and save_detections:
            with open(f'detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(self.detection_log, f, indent=2)
    
    def list_known_faces(self):
        """List all known faces in the database"""
        if not self.known_face_names:
            print("No known faces in database")
            return
        
        print(f"Known faces ({len(self.known_face_names)}):")
        for i, name in enumerate(self.known_face_names):
            print(f"{i+1}. {name}")
    
    def remove_known_face(self, name):
        """Remove a person from the known faces database"""
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            del self.known_face_names[index]
            del self.known_face_encodings[index]
            self.save_encodings()
            print(f"Removed {name} from database")
            log_face_recognition('removed', name, '', 'known', '', "Camera 0")
            return True
        else:
            print(f"{name} not found in database")
            return False

def main():
    parser = argparse.ArgumentParser(description="Enhanced Security System with Thief Detection")
    parser.add_argument("--mode", choices=["add", "detect", "monitor", "list", "remove", 
                                         "thief_add", "thief_detect", "thief_monitor", "thief_list", "thief_remove", "thief_stats"], 
                       help="Operation mode")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--name", help="Person name")
    parser.add_argument("--description", help="Description for thief")
    parser.add_argument("--threat_level", choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"], 
                       default="HIGH", help="Threat level for thief")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--save", action="store_true", help="Save detection results")
    parser.add_argument("--system", choices=["normal", "thief"], default="normal", 
                       help="Choose system type")
    
    args = parser.parse_args()
    
    # Initialize systems
    normal_system = FaceRecognitionSystem()
    thief_system = ThiefDetectionSystem()
    
    # Thief detection modes
    if args.mode == "thief_add":
        if args.image and args.name:
            description = args.description or ""
            thief_system.add_thief(args.image, args.name, description, args.threat_level)
        else:
            print("Please provide --image and --name for adding a thief")
    
    elif args.mode == "thief_detect":
        if args.image:
            results, alert = thief_system.detect_thieves_in_image(args.image, save_result=args.save)
            print(f"Thief detection results for {args.image}:")
            for thief in results:
                status = "🚨 THIEF" if thief['is_thief'] else "Person"
                print(f"- {status}: {thief['name']} (Confidence: {thief['confidence']:.2f}, Threat: {thief['threat_level']})")
            if alert:
                print("🚨 ALERT TRIGGERED! 🚨")
        else:
            print("Please provide --image for thief detection")
    
    elif args.mode == "thief_monitor":
        thief_system.start_thief_monitoring(args.camera, save_detections=args.save)
    
    elif args.mode == "thief_list":
        thief_system.list_thieves()
    
    elif args.mode == "thief_remove":
        if args.name:
            thief_system.remove_thief(args.name)
        else:
            print("Please provide --name to remove thief")
    
    elif args.mode == "thief_stats":
        stats = thief_system.get_thief_statistics()
        print("🚨 THIEF DETECTION STATISTICS 🚨")
        print("=" * 40)
        print(f"Total Thieves in Database: {stats['total_thieves']}")
        print(f"Recent Alerts Today: {stats['recent_alerts_today']}")
        print(f"Total Alerts: {stats['total_alerts']}")
        print("\nThreat Level Distribution:")
        for level, count in stats['threat_levels'].items():
            print(f"  {level}: {count}")
    
    # Normal face recognition modes
    elif args.mode == "add":
        if args.image and args.name:
            normal_system.add_known_person(args.image, args.name)
            normal_system.save_encodings()
        else:
            print("Please provide --image and --name for adding a person")
    
    elif args.mode == "detect":
        if args.image:
            results = normal_system.detect_faces_in_image(args.image, save_result=args.save)
            print(f"Detection results for {args.image}:")
            for person in results:
                print(f"- {person['name']} (Confidence: {person['confidence']:.2f})")
        else:
            print("Please provide --image for detection")
    
    elif args.mode == "monitor":
        normal_system.start_video_monitoring(args.camera, save_detections=args.save)
    
    elif args.mode == "list":
        normal_system.list_known_faces()
    
    elif args.mode == "remove":
        if args.name:
            normal_system.remove_known_face(args.name)
        else:
            print("Please provide --name to remove")
    
    else:
        # Interactive mode
        print("🚨 ENHANCED SECURITY SYSTEM 🚨")
        print("=" * 40)
        print("NORMAL FACE RECOGNITION:")
        print("1. Load faces from 'known_faces' directory")
        print("2. Add single person")
        print("3. Detect faces in image") 
        print("4. Start video monitoring")
        print("5. List known faces")
        print("6. Remove person")
        print()
        print("🚨 THIEF DETECTION SYSTEM:")
        print("7. Add thief to database")
        print("8. Detect thieves in image")
        print("9. Start thief monitoring")
        print("10. List known thieves")
        print("11. Remove thief")
        print("12. View thief statistics")
        print("13. Load thieves from directory")
        
        choice = input("Enter choice (1-13): ")
        
        if choice == "1":
            normal_system.load_faces_from_directory()
        elif choice == "2":
            image_path = input("Enter image path: ")
            name = input("Enter person name: ")
            normal_system.add_known_person(image_path, name)
            normal_system.save_encodings()
        elif choice == "3":
            image_path = input("Enter image path: ")
            results = normal_system.detect_faces_in_image(image_path, save_result=True)
            for person in results:
                print(f"Found: {person['name']} (Confidence: {person['confidence']:.2f})")
        elif choice == "4":
            normal_system.start_video_monitoring()
        elif choice == "5":
            normal_system.list_known_faces()
        elif choice == "6":
            name = input("Enter name to remove: ")
            normal_system.remove_known_face(name)
        elif choice == "7":
            image_path = input("Enter thief image path: ")
            name = input("Enter thief name: ")
            description = input("Enter description (optional): ")
            threat_level = input("Enter threat level (LOW/MEDIUM/HIGH/CRITICAL) [HIGH]: ") or "HIGH"
            thief_system.add_thief(image_path, name, description, threat_level)
        elif choice == "8":
            image_path = input("Enter image path: ")
            results, alert = thief_system.detect_thieves_in_image(image_path, save_result=True)
            for thief in results:
                status = "🚨 THIEF" if thief['is_thief'] else "Person"
                print(f"Found: {status} {thief['name']} (Confidence: {thief['confidence']:.2f})")
            if alert:
                print("🚨 ALERT TRIGGERED! 🚨")
        elif choice == "9":
            thief_system.start_thief_monitoring()
        elif choice == "10":
            thief_system.list_thieves()
        elif choice == "11":
            name = input("Enter thief name to remove: ")
            thief_system.remove_thief(name)
        elif choice == "12":
            stats = thief_system.get_thief_statistics()
            print("🚨 THIEF DETECTION STATISTICS 🚨")
            print("=" * 40)
            print(f"Total Thieves: {stats['total_thieves']}")
            print(f"Recent Alerts Today: {stats['recent_alerts_today']}")
            print(f"Total Alerts: {stats['total_alerts']}")
            print("\nThreat Level Distribution:")
            for level, count in stats['threat_levels'].items():
                print(f"  {level}: {count}")
        elif choice == "13":
            # Load thieves from directory
            if os.path.exists("thief_faces"):
                for filename in os.listdir("thief_faces"):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join("thief_faces", filename)
                        thief_name = os.path.splitext(filename)[0]
                        thief_system.add_thief(image_path, thief_name, f"Loaded from {filename}", "HIGH")
                print("✅ Thieves loaded from directory")
            else:
                print("❌ 'thief_faces' directory not found")

if __name__ == "__main__":
    main()