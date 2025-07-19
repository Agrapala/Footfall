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
        
        # Face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Frame processing settings
        self.frame_skip = 5  # Process every 5th frame for performance
        self.frame_count = 0
        
        self.logged_person_ids = set()
        
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
        """Process a single frame and detect humans"""
        self.frame_count += 1
        current_time = datetime.now()
        
        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            return frame, self.get_unique_count_in_period(current_time)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        current_detections = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_image = frame[y:y+h, x:x+w]
            
            # Skip very small faces
            if w < 50 or h < 50:
                continue
            
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
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
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
    
    def process_frame(self, frame):
        """Process frame for person detection"""
        current_time = datetime.now()
        
        # Detect people
        people, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(32, 32), scale=1.05
        )
        
        current_detections = []
        
        for (x, y, w, h) in people:
            # Simple tracking by overlap with previous detections
            matched = False
            best_overlap = 0
            best_id = None
            
            for track_id, track_data in self.person_tracks.items():
                if current_time - track_data['last_seen'] < timedelta(seconds=5):
                    overlap = self.calculate_overlap((x, y, w, h), track_data['last_bbox'])
                    if overlap > 0.3 and overlap > best_overlap:
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
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Starting {method_name} Human Counter")
    print("Press 'q' to quit, 's' for statistics, 'r' to reset count")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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