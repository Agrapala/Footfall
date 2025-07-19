import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
from motion_logger import log_motion

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
    """Main function to run motion tracking"""
    tracker = HumanMotionTracker()
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting motion tracking. Press 'q' to quit, 's' for summary.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, motion_data = tracker.process_frame(frame)
        
        # Display motion information on frame
        y_offset = 30
        if motion_data['pose_detected']:
            cv2.putText(processed_frame, f"Motion: {motion_data['avg_motion']:.4f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            # Display events
            for event in motion_data['events']:
                cv2.putText(processed_frame, event, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_offset += 25
        else:
            cv2.putText(processed_frame, "No pose detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Human Motion Tracking', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            summary = tracker.get_motion_summary()
            print(f"\nMotion Summary: {summary}")
            # Log motion summary
            if isinstance(summary, dict):
                log_motion(summary.get('average_motion', ''), summary.get('motion_level', ''), '', summary.get('frames_analyzed', ''), "Camera 0")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    final_summary = tracker.get_motion_summary()
    print(f"\nFinal Motion Summary: {final_summary}")
    # Log final summary
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


# Usage examples:
# 1. Run main() for advanced pose-based motion tracking
# 2. Run simple_motion_demo() for basic motion detection
# 3. Use HumanMotionTracker class in your own applications