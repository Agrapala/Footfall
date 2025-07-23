import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
from motion_logger import log_motion
import matplotlib.pyplot as plt
import collections

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
        # Use nose and shoulders for rough head direction
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        # Midpoint between shoulders
        mid_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_y = (left_shoulder.y + right_shoulder.y) / 2
        dx = nose.x - mid_x
        dy = nose.y - mid_y
        x_thresh = 0.05
        y_thresh = 0.03  # More sensitive
        # Looking down: nose below both shoulders
        if nose.y > left_shoulder.y and nose.y > right_shoulder.y:
            return "Looking down"
        if dx < -x_thresh:
            return "Looking right"   # User's right
        elif dx > x_thresh:
            return "Looking left"    # User's left
        elif dy < -y_thresh:
            return "Looking up"
        else:
            return "Looking center"

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
            'avg_motion': 0
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
                
                # Detect hand events
                hand_events = self.detect_hand_events(results.pose_landmarks)
                motion_data['events'].extend(hand_events)
                
                # Detect head direction
                head_dir = self.detect_head_direction(results.pose_landmarks)
                motion_data['events'].append(head_dir)
                
                # Store history
                self.velocity_history.append(velocities)
            
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
            'motion_level': 'High' if avg_motion > 0.1 else 'Medium' if avg_motion > 0.05 else 'Low'
        }


def main():
    """Main function to run motion tracking with enhanced output"""
    tracker = HumanMotionTracker()
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Starting motion tracking. Press 'q' to quit, 's' for session summary.")
    motion_history = collections.deque(maxlen=100)
    event_history = collections.deque(maxlen=100)
    fps_history = collections.deque(maxlen=30)
    last_events = collections.deque(maxlen=3)
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, motion_data = tracker.process_frame(frame)
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0
        prev_time = now
        fps_history.append(fps)
        # Update histories
        motion_history.append(motion_data['avg_motion'])
        for event in motion_data['events']:
            event_history.append(event)
            last_events.append(event)
        # Remove live motion graph overlay
        # Only show last 3 hand/head events
        y_offset = 30
        hand_event_colors = {
            "Left hand up": (255, 128, 0),
            "Right hand up": (0, 128, 255),
            "Both hands up": (0, 255, 255),
            "Left hand out": (255, 0, 128),
            "Right hand out": (128, 0, 255),
            "Looking left": (0, 200, 255),
            "Looking right": (0, 255, 200),
            "Looking up": (255, 200, 0),
            "Looking down": (200, 0, 255),
            "Looking center": (200, 255, 0)
        }
        shown = 0
        for event in reversed(last_events):
            if event in hand_event_colors:
                color = hand_event_colors[event]
                cv2.putText(processed_frame, event, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                y_offset += 40
                shown += 1
                if shown >= 3:
                    break
        if shown < 3:
            for event in reversed(last_events):
                if event not in hand_event_colors:
                    cv2.putText(processed_frame, event, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 30
                    shown += 1
                    if shown >= 3:
                        break
        cv2.imshow('Human Motion Tracking (Enhanced)', processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Show session summary with matplotlib
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(motion_history)
            plt.title('Motion Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Avg Motion')
            plt.subplot(1, 2, 2)
            if event_history:
                event_counts = collections.Counter(event_history)
                plt.pie(event_counts.values(), labels=event_counts.keys(), autopct='%1.1f%%')
                plt.title('Event Distribution')
            else:
                plt.text(0.5, 0.5, 'No events', ha='center', va='center')
            plt.tight_layout()
            plt.show()
    cap.release()
    cv2.destroyAllWindows()
    # Final summary
    final_summary = tracker.get_motion_summary()
    print(f"\nFinal Motion Summary: {final_summary}")
    if isinstance(final_summary, dict):
        log_motion(final_summary.get('average_motion', ''), final_summary.get('motion_level', ''), '', final_summary.get('frames_analyzed', ''), "Camera 0")


if __name__ == "__main__":
    main()


# Alternative: Simplified motion detection using background subtraction
class SimpleMotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_threshold = 1000  # Adjust based on your needs
        
    def detect_motion(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small movements
                total_area += area
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if total_area > self.motion_threshold:
            motion_detected = True
            cv2.putText(frame, f"Motion: {total_area:.0f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, motion_detected, total_area


def simple_motion_demo():
    """Demo of simple motion detection"""
    detector = SimpleMotionDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, motion_detected, motion_area = detector.detect_motion(frame)
        
        cv2.imshow('Simple Motion Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


