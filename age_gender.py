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

class GenderAgeDetector:
    def __init__(self):
        """Initialize the gender and age detection model"""
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
        
        # MediaPipe face detection
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def get_face_box(self, frame):
        # Returns list of [x1, y1, x2, y2] for each detected face using MediaPipe
        h, w = frame.shape[:2]
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_boxes = []
        if results.detections:
            for detection in results.detections:
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
    
    def predict_age_gender(self, face):
        """Predict age and gender from face ROI"""
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.model_mean, swapRB=False)
        
        # Predict gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        gender_confidence = gender_preds[0].max()
        
        # Predict age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        age_confidence = age_preds[0].max()
        
        return gender, age, gender_confidence, age_confidence
    
    def process_image(self, image_path):
        """Process a single image"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not load image: {image_path}")
            return None
            
        face_boxes = self.get_face_box(frame)
        log_file = 'age_gender_log.csv'
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['face_id', 'gender', 'age', 'bbox', 'source', 'timestamp'])
        logged_encodings = []
        SIMILARITY_THRESHOLD = 0.6
        for i, face_box in enumerate(face_boxes):
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
                    gender, age, gender_conf, age_conf = self.predict_age_gender(face)
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([i, gender, age, f'{face_box[0]},{face_box[1]},{face_box[2]},{face_box[3]}', image_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                    logged_encodings.append(encoding)
            # Draw rectangle around face (blue)
            cv2.rectangle(frame, (face_box[0], face_box[1]), 
                         (face_box[2], face_box[3]), (60, 120, 255), 2)
            # Add labels
            label = f"{gender} | {age}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # Semi-transparent black rectangle for text background
            overlay = frame.copy()
            y1 = max(face_box[1] - label_size[1] - 16, 0)
            y2 = face_box[1]
            x1 = face_box[0]
            x2 = face_box[0] + label_size[0] + 16
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # White text
            cv2.putText(frame, label, (x1 + 8, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
    
    def process_video(self, video_source=0):
        """Process video stream (webcam or video file)"""
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
                writer.writerow(['face_id', 'gender', 'age', 'bbox', 'frame', 'timestamp'])
        # Use a buffer for each face (by bbox) to smooth age
        age_buffers = defaultdict(lambda: deque(maxlen=15))
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
                        gender, age, gender_conf, age_conf = self.predict_age_gender(face)
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([i, gender, age, f'{face_box[0]},{face_box[1]},{face_box[2]},{face_box[3]}', frame_count, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                        logged_encodings.append(encoding)
                bbox_key = (int((face_box[0]+face_box[2])/2), int((face_box[1]+face_box[3])/2))
                age_num = int(age.strip('()').split('-')[0]) if '-' in age else int(age.strip('()').split('-')[0])
                age_buffers[bbox_key].append(age_num)
                smoothed_age = int(round(np.mean(age_buffers[bbox_key])))
                age_range = f"{smoothed_age-3}–{smoothed_age+3}"
                label = f"{gender} | Age: {age_range}"
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
            cv2.imshow('Gender and Age Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'detected_frame_{frame_count}.jpg', frame)
                print(f"Frame saved as detected_frame_{frame_count}.jpg")
                frame_count += 1
                
        cap.release()
        cv2.destroyAllWindows()

    def process_video_with_tracking(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return
        print("Press 'q' to quit, 's' to save current frame")
        frame_count = 0
        trackers = cv2.MultiTracker_create()
        face_id_counter = 0
        face_id_map = {}  # tracker index -> face ID
        age_buffers = defaultdict(lambda: deque(maxlen=15))
        logged_encodings = []
        SIMILARITY_THRESHOLD = 0.6
        logged_ids = set()
        log_file = 'age_gender_log.csv'
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['face_id', 'gender', 'age_range', 'bbox', 'frame', 'timestamp'])
        tracker_boxes = []  # Store last known boxes for IoU matching
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            # Every 30 frames or if no trackers, run face detection
            if frame_count % 30 == 0 or len(trackers.getObjects()) == 0:
                face_boxes = self.get_face_box(frame)
                # Match detected faces to existing trackers using IoU
                new_trackers = cv2.MultiTracker_create()
                new_face_id_map = {}
                used_ids = set()
                for box in face_boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    best_iou = 0
                    best_idx = -1
                    for idx, tbox in enumerate(tracker_boxes):
                        iou_val = iou((x1, y1, x2, y2), tbox)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_idx = idx
                    if best_iou > 0.3 and best_idx in face_id_map and face_id_map[best_idx] not in used_ids:
                        # Match found, keep same ID
                        tracker = cv2.TrackerCSRT_create()
                        new_trackers.add(tracker, frame, (x1, y1, w, h))
                        new_face_id_map[len(new_face_id_map)] = face_id_map[best_idx]
                        used_ids.add(face_id_map[best_idx])
                    else:
                        # New face, assign new ID
                        tracker = cv2.TrackerCSRT_create()
                        new_trackers.add(tracker, frame, (x1, y1, w, h))
                        new_face_id_map[len(new_face_id_map)] = face_id_counter
                        face_id_counter += 1
                trackers = new_trackers
                face_id_map = new_face_id_map
                tracker_boxes = face_boxes
            # Update trackers
            success, boxes = trackers.update(frame)
            tracker_boxes = []
            for i, box in enumerate(boxes):
                x, y, w, h = [int(v) for v in box]
                tracker_boxes.append([x, y, x+w, y+h])
                face_img = frame[y:y+h, x:x+w]
                if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    continue
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
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
                        gender, age, gender_conf, age_conf = self.predict_age_gender(face_img)
                        bbox_key = (int(x + w/2), int(y + h/2))
                        age_num = int(age.strip('()').split('-')[0]) if '-' in age else int(age.strip('()').split('-')[0])
                        age_buffers[bbox_key].append(age_num)
                        smoothed_age = int(round(np.mean(age_buffers[bbox_key])))
                        age_range = f"{smoothed_age-3}–{smoothed_age+3}"
                        face_id = face_id_map.get(i, -1)
                        label = f"ID:{face_id} | {gender} | Age: {age_range}"
                        # Draw rectangle around face (blue)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 120, 255), 2)
                        # Add labels
                        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        overlay = frame.copy()
                        y1 = max(y - label_size[1] - 16, 0)
                        y2 = y
                        x1 = x
                        x2 = x + label_size[0] + 16
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                        alpha = 0.6
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        cv2.putText(frame, label, (x1 + 8, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        # Log only once per unique face_id
                        if face_id not in logged_ids:
                            with open(log_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([face_id, gender, age_range, f'{x},{y},{x+w},{y+h}', frame_count, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                            logged_ids.add(face_id)
                            logged_encodings.append(encoding)
            cv2.imshow('Gender and Age Detection (Tracking)', frame)
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
    parser = argparse.ArgumentParser(description='Gender and Age Detection')
    parser.add_argument('--input', type=str, help='Path to input image or video file')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam', 'track'], 
                       default='webcam', help='Detection mode')
    parser.add_argument('--download', action='store_true', 
                       help='Download required model files')
    
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
    detector = GenderAgeDetector()
    
    if args.mode == 'image':
        if not args.input:
            print("Please provide input image path with --input")
            return
        
        result = detector.process_image(args.input)
        if result is not None:
            cv2.imshow('Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save result
            output_path = f"result_{os.path.basename(args.input)}"
            cv2.imwrite(output_path, result)
            print(f"Result saved as {output_path}")
            
    elif args.mode == 'video':
        if not args.input:
            print("Please provide input video path with --input")
            return
        detector.process_video(args.input)
        
    elif args.mode == 'track':
        detector.process_video_with_tracking(0)
        
    else:  # webcam
        detector.process_video(0)

if __name__ == "__main__":
    main()