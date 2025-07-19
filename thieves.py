import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
import argparse
import json
from face_recognition_logger import log_face_recognition

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
    parser = argparse.ArgumentParser(description="Face Recognition Security System")
    parser.add_argument("--mode", choices=["add", "detect", "monitor", "list", "remove"], 
                       help="Operation mode")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--name", help="Person name")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--save", action="store_true", help="Save detection results")
    
    args = parser.parse_args()
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    if args.mode == "add":
        if args.image and args.name:
            system.add_known_person(args.image, args.name)
            system.save_encodings()
        else:
            print("Please provide --image and --name for adding a person")
    
    elif args.mode == "detect":
        if args.image:
            results = system.detect_faces_in_image(args.image, save_result=args.save)
            print(f"Detection results for {args.image}:")
            for person in results:
                print(f"- {person['name']} (Confidence: {person['confidence']:.2f})")
        else:
            print("Please provide --image for detection")
    
    elif args.mode == "monitor":
        system.start_video_monitoring(args.camera, save_detections=args.save)
    
    elif args.mode == "list":
        system.list_known_faces()
    
    elif args.mode == "remove":
        if args.name:
            system.remove_known_face(args.name)
        else:
            print("Please provide --name to remove")
    
    else:
        # Interactive mode
        print("Face Recognition Security System")
        print("1. Load faces from 'known_faces' directory")
        print("2. Add single person")
        print("3. Detect faces in image") 
        print("4. Start video monitoring")
        print("5. List known faces")
        print("6. Remove person")
        
        choice = input("Enter choice (1-6): ")
        
        if choice == "1":
            system.load_faces_from_directory()
        elif choice == "2":
            image_path = input("Enter image path: ")
            name = input("Enter person name: ")
            system.add_known_person(image_path, name)
            system.save_encodings()
        elif choice == "3":
            image_path = input("Enter image path: ")
            results = system.detect_faces_in_image(image_path, save_result=True)
            for person in results:
                print(f"Found: {person['name']} (Confidence: {person['confidence']:.2f})")
        elif choice == "4":
            system.start_video_monitoring()
        elif choice == "5":
            system.list_known_faces()
        elif choice == "6":
            name = input("Enter name to remove: ")
            system.remove_known_face(name)

if __name__ == "__main__":
    main()