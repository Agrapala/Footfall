import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
from motion_logger import log_motion
import matplotlib.pyplot as plt
import collections
import os
import json
from datetime import datetime
import threading
import queue

class HumanMotionTracker:
    def __init__(self, max_history=30):
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Motion tracking variables
        self.max_history = max_history
        self.pose_history = deque(maxlen=max_history)
        self.velocity_history = deque(maxlen=max_history)
        self.prev_landmarks = None
        self.prev_time = None
        
        # NEW: Activity recognition
        self.activity_buffer = deque(maxlen=60)  # 2 seconds at 30fps
        self.current_activity = "Unknown"
        
        # NEW: Zone monitoring
        self.zones = {}
        self.zone_violations = []
        
        # NEW: Gesture recognition
        self.gesture_buffer = deque(maxlen=15)  # 0.5 seconds
        self.recognized_gestures = []
        
        # NEW: Posture analysis
        self.posture_alerts = []
        self.slouch_counter = 0
        
        # NEW: Fall detection
        self.fall_detected = False
        self.fall_threshold = 0.3  # Angle threshold for fall detection
        
        # NEW: Energy expenditure tracking
        self.energy_log = []
        self.calories_burned = 0.0
        
        # NEW: Multi-person tracking
        self.person_tracker = {}
        self.next_person_id = 0
        
        # NEW: Real-time analytics
        self.analytics_queue = queue.Queue()
        self.analytics_thread = None
        
        # Key body parts for motion analysis
        self.key_points = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def calculate_velocity(self, curr_landmarks, prev_landmarks, time_diff):
        """Calculate velocity for each key point"""
        if time_diff == 0:
            return {}
        
        velocities = {}
        for point in self.key_points:
            curr_point = curr_landmarks.landmark[point]
            prev_point = prev_landmarks.landmark[point]
            
            distance = self.calculate_distance(curr_point, prev_point)
            velocity = distance / time_diff
            velocities[point.name] = velocity
            
        return velocities

    def calculate_acceleration(self, curr_velocity, prev_velocity, time_diff):
        """Calculate acceleration for each key point"""
        if time_diff == 0 or not prev_velocity:
            return {}
        
        accelerations = {}
        for point_name in curr_velocity:
            if point_name in prev_velocity:
                acc = (curr_velocity[point_name] - prev_velocity[point_name]) / time_diff
                accelerations[point_name] = acc
            else:
                accelerations[point_name] = 0
                
        return accelerations

    def detect_motion_events(self, velocities, threshold=0.1):
        """Detect significant motion events"""
        events = []
        
        # Overall body motion
        avg_velocity = sum(velocities.values()) / len(velocities) if velocities else 0
        if avg_velocity > threshold:
            events.append(f"Active motion detected (avg vel: {avg_velocity:.3f})")
        
        # Specific body part analysis
        arm_points = ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW']
        arm_velocity = sum(velocities.get(p, 0) for p in arm_points) / len(arm_points)
        
        leg_points = ['LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_KNEE', 'RIGHT_KNEE']
        leg_velocity = sum(velocities.get(p, 0) for p in leg_points) / len(leg_points)
        
        if arm_velocity > threshold * 2:
            events.append("High arm activity")
        if leg_velocity > threshold * 2:
            events.append("High leg activity")
            
        return events

    def detect_hand_events(self, landmarks):
        events = []
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Hand up: wrist higher (y is smaller) than shoulder
        if left_wrist.y < left_shoulder.y:
            events.append("Left hand up")
        if right_wrist.y < right_shoulder.y:
            events.append("Right hand up")
        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            events.append("Both hands up")
        
        # Hand out: wrist x much less/greater than shoulder x
        if left_wrist.x < left_shoulder.x - 0.15:
            events.append("Left hand out")
        if right_wrist.x > right_shoulder.x + 0.15:
            events.append("Right hand out")
        
        return events

    def detect_head_direction(self, landmarks):
        """Use nose and shoulders for rough head direction"""
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Midpoint between shoulders
        mid_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_y = (left_shoulder.y + right_shoulder.y) / 2
        dx = nose.x - mid_x
        dy = nose.y - mid_y
        x_thresh = 0.05
        y_thresh = 0.03
        
        # Looking down: nose below both shoulders
        if nose.y > left_shoulder.y and nose.y > right_shoulder.y:
            return "Looking down"
        if dx < -x_thresh:
            return "Looking right"
        elif dx > x_thresh:
            return "Looking left"
        elif dy < -y_thresh:
            return "Looking up"
        else:
            return "Looking center"

    # NEW FEATURE 1: Activity Recognition
    def recognize_activity(self, landmarks, velocities):
        """Recognize current activity based on pose and motion patterns"""
        if not velocities:
            return "Stationary"
        
        avg_velocity = sum(velocities.values()) / len(velocities)
        
        # Get key joint positions
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Activity classification logic
        leg_velocity = sum([velocities.get('LEFT_ANKLE', 0), velocities.get('RIGHT_ANKLE', 0),
                           velocities.get('LEFT_KNEE', 0), velocities.get('RIGHT_KNEE', 0)]) / 4
        
        arm_velocity = sum([velocities.get('LEFT_WRIST', 0), velocities.get('RIGHT_WRIST', 0),
                           velocities.get('LEFT_ELBOW', 0), velocities.get('RIGHT_ELBOW', 0)]) / 4
        
        # Simple activity classification
        if avg_velocity < 0.02:
            activity = "Standing/Sitting"
        elif leg_velocity > 0.1 and avg_velocity > 0.08:
            if left_ankle.y > left_knee.y - 0.1 and right_ankle.y > right_knee.y - 0.1:
                activity = "Walking"
            else:
                activity = "Running"
        elif arm_velocity > 0.15:
            activity = "Exercising"
        elif avg_velocity > 0.05:
            activity = "General Movement"
        else:
            activity = "Stationary"
        
        self.activity_buffer.append(activity)
        
        # Smooth activity recognition
        if len(self.activity_buffer) > 10:
            activity_counts = collections.Counter(list(self.activity_buffer)[-10:])
            self.current_activity = activity_counts.most_common(1)[0][0]
        
        return self.current_activity

    # NEW FEATURE 2: Zone Monitoring
    def define_zone(self, name, x1, y1, x2, y2):
        """Define a monitoring zone in the frame"""
        self.zones[name] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'violations': 0}

    def check_zone_violations(self, landmarks, frame_shape):
        """Check if person enters restricted zones"""
        violations = []
        
        # Use center of body (average of shoulders and hips)
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        # Convert to pixel coordinates
        pixel_x = int(center_x * frame_shape[1])
        pixel_y = int(center_y * frame_shape[0])
        
        for zone_name, zone in self.zones.items():
            if (zone['x1'] <= pixel_x <= zone['x2'] and 
                zone['y1'] <= pixel_y <= zone['y2']):
                violations.append(f"Zone violation: {zone_name}")
                self.zones[zone_name]['violations'] += 1
        
        return violations

    # NEW FEATURE 3: Gesture Recognition
    def recognize_gestures(self, landmarks):
        """Recognize specific gestures"""
        gestures = []
        
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        # Wave gesture (hand moving side to side above shoulder)
        if right_wrist.y < right_shoulder.y - 0.1:
            self.gesture_buffer.append('right_hand_up')
        if left_wrist.y < left_shoulder.y - 0.1:
            self.gesture_buffer.append('left_hand_up')
        
        # Check for wave pattern in buffer
        if len(self.gesture_buffer) >= 10:
            recent = list(self.gesture_buffer)[-10:]
            if recent.count('right_hand_up') >= 7:
                gestures.append("Right hand wave")
            if recent.count('left_hand_up') >= 7:
                gestures.append("Left hand wave")
        
        # Clapping gesture (both hands close together in front of body)
        if (abs(left_wrist.x - right_wrist.x) < 0.1 and 
            abs(left_wrist.y - right_wrist.y) < 0.1 and
            left_wrist.y > nose.y + 0.1):
            gestures.append("Clapping")
        
        return gestures

    # NEW FEATURE 4: Posture Analysis
    def analyze_posture(self, landmarks):
        """Analyze posture and detect poor posture"""
        alerts = []
        
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate shoulder alignment
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_slope > 0.05:
            alerts.append("Uneven shoulders")
        
        # Calculate forward head posture
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        head_forward = nose.x - shoulder_center_x
        if abs(head_forward) > 0.08:
            alerts.append("Forward head posture")
            self.slouch_counter += 1
        else:
            self.slouch_counter = max(0, self.slouch_counter - 1)
        
        # Prolonged poor posture warning
        if self.slouch_counter > 150:  # 5 seconds at 30fps
            alerts.append("PROLONGED POOR POSTURE")
        
        return alerts

    # NEW FEATURE 5: Fall Detection
    def detect_fall(self, landmarks):
        """Detect if person has fallen"""
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate body orientation
        hip_center_y = (left_hip.y + right_hip.y) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2
        
        # If hips are close to ankle level, possible fall
        body_height = abs(nose.y - ankle_center_y)
        if body_height < 0.3:  # Very compressed body
            self.fall_detected = True
            return True
        else:
            self.fall_detected = False
            return False

    # NEW FEATURE 6: Energy Expenditure Estimation
    def estimate_energy_expenditure(self, velocities, time_diff):
        """Estimate energy expenditure based on movement intensity"""
        if not velocities or time_diff == 0:
            return 0
        
        avg_velocity = sum(velocities.values()) / len(velocities)
        
        # Simple METs (Metabolic Equivalent) calculation
        if avg_velocity > 0.15:
            mets = 6.0  # Running
        elif avg_velocity > 0.08:
            mets = 4.0  # Walking brisk
        elif avg_velocity > 0.04:
            mets = 3.0  # Walking moderate
        else:
            mets = 1.5  # Standing/light activity
        
        # Calories = METs * weight(kg) * time(hours)
        # Assuming 70kg person
        weight_kg = 70
        time_hours = time_diff / 3600
        calories = mets * weight_kg * time_hours
        
        self.calories_burned += calories
        self.energy_log.append({
            'timestamp': time.time(),
            'mets': mets,
            'calories': calories,
            'activity': self.current_activity
        })
        
        return calories

    # NEW FEATURE 7: Data Export and Analytics
    def export_session_data(self, filename=None):
        """Export session data to JSON file"""
        if filename is None:
            filename = f"motion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'total_calories': self.calories_burned,
            'energy_log': self.energy_log,
            'zone_violations': {name: zone['violations'] for name, zone in self.zones.items()},
            'recognized_gestures': self.recognized_gestures,
            'posture_alerts': self.posture_alerts,
            'fall_incidents': self.fall_detected,
            'session_summary': self.get_motion_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return filename

    # NEW FEATURE 8: Real-time Alerts
    def check_alerts(self, motion_data):
        """Check for various alert conditions"""
        alerts = []
        
        # High activity alert
        if motion_data['avg_motion'] > 0.2:
            alerts.append("HIGH_ACTIVITY")
        
        # Inactivity alert
        if len(self.velocity_history) > 90:  # 3 seconds
            recent_motion = list(self.velocity_history)[-90:]
            avg_recent = sum(sum(v.values())/len(v) for v in recent_motion) / len(recent_motion)
            if avg_recent < 0.01:
                alerts.append("PROLONGED_INACTIVITY")
        
        # Fall alert
        if self.fall_detected:
            alerts.append("FALL_DETECTED")
        
        return alerts

    def process_frame(self, frame):
        """Process a single frame and return motion data"""
        current_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        motion_data = {
            'pose_detected': False,
            'velocities': {},
            'accelerations': {},
            'events': [],
            'avg_motion': 0,
            'activity': 'Unknown',
            'zone_violations': [],
            'gestures': [],
            'posture_alerts': [],
            'fall_detected': False,
            'energy_expenditure': 0,
            'alerts': []
        }
        
        if results.pose_landmarks:
            motion_data['pose_detected'] = True
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Calculate motion metrics
            if self.prev_landmarks and self.prev_time:
                time_diff = current_time - self.prev_time
                
                # Calculate velocities
                velocities = self.calculate_velocity(
                    results.pose_landmarks, self.prev_landmarks, time_diff)
                motion_data['velocities'] = velocities
                
                # Calculate accelerations
                if len(self.velocity_history) > 0:
                    prev_velocities = self.velocity_history[-1]
                    accelerations = self.calculate_acceleration(
                        velocities, prev_velocities, time_diff)
                    motion_data['accelerations'] = accelerations
                
                # Detect motion events
                motion_data['events'] = self.detect_motion_events(velocities)
                motion_data['avg_motion'] = sum(velocities.values()) / len(velocities)
                
                # NEW: Activity recognition
                motion_data['activity'] = self.recognize_activity(results.pose_landmarks, velocities)
                
                # NEW: Zone monitoring
                motion_data['zone_violations'] = self.check_zone_violations(results.pose_landmarks, frame.shape)
                
                # NEW: Energy expenditure
                motion_data['energy_expenditure'] = self.estimate_energy_expenditure(velocities, time_diff)
                
                # Store history
                self.velocity_history.append(velocities)
            
            # Detect hand events
            hand_events = self.detect_hand_events(results.pose_landmarks)
            motion_data['events'].extend(hand_events)
            
            # Detect head direction
            head_dir = self.detect_head_direction(results.pose_landmarks)
            motion_data['events'].append(head_dir)
            
            # NEW: Gesture recognition
            motion_data['gestures'] = self.recognize_gestures(results.pose_landmarks)
            self.recognized_gestures.extend(motion_data['gestures'])
            
            # NEW: Posture analysis
            motion_data['posture_alerts'] = self.analyze_posture(results.pose_landmarks)
            self.posture_alerts.extend(motion_data['posture_alerts'])
            
            # NEW: Fall detection
            motion_data['fall_detected'] = self.detect_fall(results.pose_landmarks)
            
            # NEW: Real-time alerts
            motion_data['alerts'] = self.check_alerts(motion_data)
            
            # Update previous state
            self.prev_landmarks = results.pose_landmarks
            self.prev_time = current_time
            self.pose_history.append(results.pose_landmarks)
        
        return frame, motion_data

    def get_motion_summary(self):
        """Get summary of motion over time"""
        if not self.velocity_history:
            return "No motion data available"
        
        # Calculate average motion over time
        total_motion = 0
        frame_count = len(self.velocity_history)
        
        for velocities in self.velocity_history:
            total_motion += sum(velocities.values()) / len(velocities)
        
        avg_motion = total_motion / frame_count if frame_count > 0 else 0
        
        return {
            'average_motion': avg_motion,
            'frames_analyzed': frame_count,
            'motion_level': 'High' if avg_motion > 0.1 else 'Medium' if avg_motion > 0.05 else 'Low',
            'total_calories_burned': self.calories_burned,
            'primary_activity': self.current_activity,
            'total_gestures': len(self.recognized_gestures),
            'posture_issues': len(self.posture_alerts),
            'fall_incidents': 1 if self.fall_detected else 0
        }


def main():
    """Main function to run enhanced motion tracking"""
    tracker = HumanMotionTracker()
    
    # Define monitoring zones (example)
    tracker.define_zone("Restricted Area", 100, 100, 200, 200)
    tracker.define_zone("Danger Zone", 400, 300, 500, 400)
    
    # Use camera or video file
    video_path = ""
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        print(f"Using video file: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        print("Using webcam")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Enhanced Motion Tracking Started")
    print("Press 'q' to quit, 's' for session summary, 'e' to export data")
    
    motion_history = collections.deque(maxlen=100)
    event_history = collections.deque(maxlen=100)
    alert_log = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        processed_frame, motion_data = tracker.process_frame(frame)
        
        # Draw zones
        for zone_name, zone in tracker.zones.items():
            cv2.rectangle(processed_frame, (zone['x1'], zone['y1']), 
                         (zone['x2'], zone['y2']), (0, 0, 255), 2)
            cv2.putText(processed_frame, zone_name, (zone['x1'], zone['y1']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display enhanced information
        y_offset = 30
        
        # Activity and calories
        cv2.putText(processed_frame, f"Activity: {motion_data['activity']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25
        
        cv2.putText(processed_frame, f"Calories: {tracker.calories_burned:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25
        
        # Alerts
        for alert in motion_data['alerts']:
            cv2.putText(processed_frame, f"ALERT: {alert}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30
            alert_log.append({'time': time.time(), 'alert': alert})
        
        # Zone violations
        for violation in motion_data['zone_violations']:
            cv2.putText(processed_frame, violation, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 25
        
        # Gestures
        for gesture in motion_data['gestures']:
            cv2.putText(processed_frame, f"Gesture: {gesture}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            y_offset += 25
        
        # Posture alerts
        for posture_alert in motion_data['posture_alerts']:
            cv2.putText(processed_frame, f"Posture: {posture_alert}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            y_offset += 25
        
        cv2.imshow('Enhanced Human Motion Tracking', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Show session summary
            summary = tracker.get_motion_summary()
            print("\n=== SESSION SUMMARY ===")
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("=" * 25)
        elif key == ord('e'):
            # Export session data
            filename = tracker.export_session_data()
            print(f"Session data exported to: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary and export
    final_summary = tracker.get_motion_summary()
    print(f"\nFinal Motion Summary: {final_summary}")
    
    # Auto-export session data
    export_file = tracker.export_session_data()
    print(f"Session data automatically exported to: {export_file}")


if __name__ == "__main__":
    main()