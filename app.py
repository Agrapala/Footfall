from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
import cv2
import numpy as np
import threading
import time
import json
import os
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import sqlite3
from werkzeug.utils import secure_filename
import face_recognition

# Import our custom modules
from count import UniqueHumanCounter, SimplePersonCounter, LivenessDetector
from thieves import ThiefDetectionSystem

class CombinedSecuritySystem:
    """Combined security system integrating human counting and thief detection"""
    
    def __init__(self):
        # Initialize human counting system
        try:
            self.human_counter = UniqueHumanCounter(time_window_minutes=30)
            self.human_detection_active = True
        except ImportError:
            self.human_counter = SimplePersonCounter(time_window_minutes=30)
            self.human_detection_active = True
        
        # Initialize thief detection system
        self.thief_system = ThiefDetectionSystem()
        self.thief_detection_active = True
        
        # Combined statistics
        self.total_detections = 0
        self.thief_alerts = 0
        
        # Load existing human detection data from CSV
        self.load_human_detections_from_csv()
        
    def process_frame(self, frame):
        """Process frame with both human counting and thief detection"""
        processed_frame = frame.copy()
        human_count = 0
        thief_detected = False
        detection_info = {
            'humans': 0,
            'thieves': [],
            'alerts': [],
            'gender_age_data': []
        }
        
        # Human counting
        if self.human_detection_active and self.human_counter:
            try:
                processed_frame, human_count = self.human_counter.process_frame(frame)
                detection_info['humans'] = human_count
                
                # Add human count overlay
                cv2.putText(processed_frame, f"Humans: {human_count}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Get detailed human detection data for database storage
                # Read from CSV file to get gender and age data
                self.load_human_detections_from_csv()
                
                # Also check detection history if available
                if hasattr(self.human_counter, 'detection_history') and self.human_counter.detection_history:
                    # Get the most recent detections
                    recent_detections = list(self.human_counter.detection_history)[-5:]  # Last 5 detections
                    for detection in recent_detections:
                        if detection.get('timestamp'):
                            # Save to database
                            self.save_human_detection_to_db(detection)
                            
                            # Add to detection info for display
                            detection_info['gender_age_data'].append({
                                'person_id': detection.get('person_id', 0),
                                'timestamp': detection.get('timestamp'),
                                'gender': detection.get('gender', 'Unknown'),
                                'age': detection.get('age', 'Unknown'),
                                'confidence': detection.get('confidence', 0.0)
                            })
                
            except Exception as e:
                log_event("ERROR", f"Human counting error: {str(e)}", "ERROR")
        
        # Thief detection
        if self.thief_detection_active and self.thief_system:
            try:
                # Detect faces for thief recognition (no lighting enhancement needed)
                face_locations = face_recognition.face_locations(processed_frame)
                face_encodings = face_recognition.face_encodings(processed_frame, face_locations)
                
                # Check each face against thief database
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(self.thief_system.thief_encodings, face_encoding)
                    name = "Unknown"
                    confidence = 0
                    threat_level = "UNKNOWN"
                    
                    if len(self.thief_system.thief_encodings) > 0:
                        face_distances = face_recognition.face_distance(self.thief_system.thief_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                            name = self.thief_system.thief_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            threat_level = self.thief_system.thief_metadata[best_match_index].get('threat_level', 'HIGH')
                            thief_detected = True
                            
                            # Draw thief alert
                            cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 0, 255), 4)
                            cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            cv2.putText(processed_frame, f"🚨 THIEF: {name}", (left + 6, bottom - 6), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(processed_frame, f"THREAT: {threat_level}", (left + 6, bottom - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            
                            # Log thief detection
                            self.thief_system.log_thief_detection(name, confidence, (top, right, bottom, left))
                            log_event("THIEF_ALERT", f"Thief detected: {name} (Confidence: {confidence:.2f})", "CRITICAL")
                            
                            detection_info['thieves'].append({
                                'name': name,
                                'confidence': confidence,
                                'threat_level': threat_level,
                                'location': (top, right, bottom, left)
                            })
                            detection_info['alerts'].append(f"THIEF ALERT: {name}")
                            self.thief_alerts += 1
                        else:
                            # Draw normal person detection
                            cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(processed_frame, "Person", (left, top-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Draw normal person detection when no thief database
                        cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(processed_frame, "Person", (left, top-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
            except Exception as e:
                log_event("ERROR", f"Thief detection error: {str(e)}", "ERROR")
        
        # Add alert overlay if thief detected
        if thief_detected:
            cv2.putText(processed_frame, "🚨 THIEF DETECTED! 🚨", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(processed_frame, "ALERT: SECURITY BREACH", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        self.total_detections += 1
        return processed_frame, detection_info
    
    def get_combined_stats(self):
        """Get combined statistics from both systems"""
        stats = {
            'human_count': 0,
            'total_persons': 0,
            'time_window_minutes': 30,
            'active_persons': 0,
            'thieves_in_db': 0,
            'recent_alerts': 0,
            'total_alerts': 0,
            'threat_levels': {},
            'total_detections': self.total_detections,
            'thief_alerts': self.thief_alerts
        }
        
        # Human counting stats
        if self.human_counter:
            try:
                human_stats = self.human_counter.get_statistics()
                stats['human_count'] = human_stats['unique_count_in_window']
                stats['total_persons'] = human_stats['total_persons_ever_seen']
                stats['time_window_minutes'] = human_stats['time_window_minutes']
                stats['active_persons'] = len(human_stats.get('active_persons', []))
            except:
                pass
        
        # Thief detection stats
        if self.thief_system:
            try:
                thief_stats = self.thief_system.get_thief_statistics()
                stats['thieves_in_db'] = thief_stats['total_thieves']
                stats['recent_alerts'] = thief_stats['recent_alerts_today']
                stats['total_alerts'] = thief_stats['total_alerts']
                stats['threat_levels'] = thief_stats['threat_levels']
            except:
                pass
        
        return stats
    
    def add_thief(self, image_path, thief_name, description="", threat_level="HIGH"):
        """Add thief to the system"""
        if self.thief_system:
            return self.thief_system.add_thief(image_path, thief_name, description, threat_level)
        return False
    
    def remove_thief(self, name):
        """Remove thief from the system"""
        if self.thief_system:
            return self.thief_system.remove_thief(name)
        return False
    
    def list_thieves(self):
        """List all thieves in the system"""
        if self.thief_system:
            return self.thief_system.list_thieves()
        return []
    
    def get_thief_data(self):
        """Get thief data for web interface"""
        if self.thief_system:
            thieves = []
            for i, (name, metadata) in enumerate(zip(self.thief_system.thief_names, self.thief_system.thief_metadata)):
                thieves.append({
                    'id': i,
                    'name': name,
                    'threat_level': metadata.get('threat_level', 'UNKNOWN'),
                    'description': metadata.get('description', ''),
                    'date_added': metadata.get('date_added', '')
                })
            return thieves
        return []
    
    def save_human_detection_to_db(self, detection):
        """Save human detection data to database"""
        try:
            conn = sqlite3.connect('security_system.db')
            cursor = conn.cursor()
            
            # Check if this detection already exists (avoid duplicates)
            cursor.execute('''
                SELECT id FROM human_detections 
                WHERE person_id = ? AND timestamp = ? AND bbox = ?
            ''', (
                detection.get('person_id', 0),
                detection.get('timestamp'),
                str(detection.get('bbox', ''))
            ))
            
            if not cursor.fetchone():  # Only insert if not already exists
                cursor.execute('''
                    INSERT INTO human_detections 
                    (timestamp, person_id, gender, age, confidence, bbox, is_first_detection)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection.get('timestamp'),
                    detection.get('person_id', 0),
                    detection.get('gender', 'Unknown'),
                    detection.get('age', 'Unknown'),
                    detection.get('confidence', 0.0),
                    str(detection.get('bbox', '')),
                    detection.get('is_first_detection', False)
                ))
                
                conn.commit()
                log_event("HUMAN_DETECTION", f"Saved detection for person {detection.get('person_id', 0)}", "INFO")
            
            conn.close()
        except Exception as e:
            log_event("ERROR", f"Failed to save human detection: {str(e)}", "ERROR")
    
    def load_human_detections_from_csv(self):
        """Load human detections from CSV file and save to database"""
        try:
            import csv
            csv_file = 'human_detection_data.csv'
            
            if not os.path.exists(csv_file):
                return
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse the CSV data
                    detection = {
                        'timestamp': row.get('datetime', ''),
                        'person_id': int(row.get('person_id', 0)),
                        'gender': row.get('gender', 'Unknown'),
                        'age': row.get('age', 'Unknown'),
                        'confidence': float(row.get('confidence', 0.0)),
                        'bbox': row.get('bbox', ''),
                        'is_first_detection': row.get('first_detection', 'False').lower() == 'true'
                    }
                    
                    # Save to database if not already exists
                    self.save_human_detection_to_db(detection)
                    
        except Exception as e:
            log_event("ERROR", f"Failed to load human detections from CSV: {str(e)}", "ERROR")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Global variables for camera streams
camera_active = False
current_frame = None
security_system = None  # Combined system
camera_thread = None
camera = None

# Database setup
def init_db():
    conn = sqlite3.connect('security_system.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS human_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            person_id INTEGER,
            gender TEXT,
            age INTEGER,
            confidence REAL,
            bbox TEXT,
            is_first_detection BOOLEAN
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS thief_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            thief_name TEXT,
            confidence REAL,
            threat_level TEXT,
            location TEXT,
            camera_id TEXT,
            alert_level TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            event_type TEXT,
            message TEXT,
            severity TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def log_event(event_type, message, severity="INFO"):
    conn = sqlite3.connect('security_system.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO system_logs (timestamp, event_type, message, severity)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now(), event_type, message, severity))
    conn.commit()
    conn.close()

def camera_worker():
    global camera_active, current_frame, security_system, camera
    
    try:
        camera = cv2.VideoCapture(0)
        
        # Optimize camera settings for balanced lighting
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Allow some auto exposure
        camera.set(cv2.CAP_PROP_EXPOSURE, -3)  # Less aggressive exposure
        camera.set(cv2.CAP_PROP_BRIGHTNESS, 60)  # Slightly higher brightness
        camera.set(cv2.CAP_PROP_CONTRAST, 50)
        camera.set(cv2.CAP_PROP_SATURATION, 50)
        camera.set(cv2.CAP_PROP_GAIN, 0)
        camera.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        log_event("CAMERA", "Camera initialized", "INFO")
        
        while camera_active:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Process frame with combined security system
            if security_system:
                try:
                    processed_frame, detection_info = security_system.process_frame(frame)
                    current_frame = processed_frame.copy()
                except Exception as e:
                    log_event("ERROR", f"Security system error: {str(e)}", "ERROR")
                    current_frame = frame.copy()
            else:
                current_frame = frame.copy()
            
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        log_event("ERROR", f"Camera worker error: {str(e)}", "ERROR")
    finally:
        if camera:
            camera.release()

def generate_frames():
    global current_frame
    while camera_active:
        if current_frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, security_system, camera_thread
    
    try:
        if not camera_active:
            # Initialize combined security system
            security_system = CombinedSecuritySystem()
            
            camera_active = True
            camera_thread = threading.Thread(target=camera_worker)
            camera_thread.daemon = True
            camera_thread.start()
            
            log_event("SYSTEM", "Combined security system started", "INFO")
            return jsonify({'status': 'success', 'message': 'Security system started successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Security system already running'})
    except Exception as e:
        log_event("ERROR", f"Failed to start security system: {str(e)}", "ERROR")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, camera_thread, camera
    
    try:
        camera_active = False
        if camera:
            camera.release()
        if camera_thread:
            camera_thread.join(timeout=2)
        
        log_event("SYSTEM", "Camera stopped", "INFO")
        return jsonify({'status': 'success', 'message': 'Camera stopped successfully'})
    except Exception as e:
        log_event("ERROR", f"Failed to stop camera: {str(e)}", "ERROR")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_stats')
def get_stats():
    try:
        if security_system:
            stats = security_system.get_combined_stats()
        else:
            stats = {
                'human_count': 0,
                'total_persons': 0,
                'time_window_minutes': 30,
                'active_persons': 0,
                'thieves_in_db': 0,
                'recent_alerts': 0,
                'total_alerts': 0,
                'threat_levels': {},
                'total_detections': 0,
                'thief_alerts': 0
            }
        
        stats['camera_active'] = camera_active
        stats['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(stats)
    except Exception as e:
        log_event("ERROR", f"Failed to get stats: {str(e)}", "ERROR")
        return jsonify({'error': str(e)})

@app.route('/thieves')
def thieves_page():
    return render_template('thieves.html')

@app.route('/api/thieves', methods=['GET'])
def get_thieves():
    try:
        if security_system:
            return jsonify(security_system.get_thief_data())
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/thieves', methods=['POST'])
def add_thief():
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        threat_level = data.get('threat_level', 'HIGH')
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        if security_system:
            # This is a simplified version - you'd need to handle image upload
            return jsonify({'error': 'Image upload not implemented yet. Use command line interface for now.'}), 400
        else:
            return jsonify({'error': 'Security system not initialized'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/thieves/<int:thief_id>', methods=['DELETE'])
def remove_thief(thief_id):
    try:
        if security_system:
            thief_data = security_system.get_thief_data()
            if 0 <= thief_id < len(thief_data):
                thief_name = thief_data[thief_id]['name']
                success = security_system.remove_thief(thief_name)
                if success:
                    log_event("THIEF", f"Thief removed: {thief_name}", "INFO")
                    return jsonify({'status': 'success'})
                else:
                    return jsonify({'error': 'Failed to remove thief'}), 500
            else:
                return jsonify({'error': 'Thief not found'}), 404
        else:
            return jsonify({'error': 'Security system not initialized'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs')
def logs_page():
    return render_template('logs.html')

@app.route('/api/logs')
def get_logs():
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Get recent logs
        cursor.execute('''
            SELECT timestamp, event_type, message, severity
            FROM system_logs
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        
        logs = []
        for row in cursor.fetchall():
            logs.append({
                'timestamp': row[0],
                'event_type': row[1],
                'message': row[2],
                'severity': row[3]
            })
        
        conn.close()
        return jsonify(logs)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/human_detections')
def get_human_detections():
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Get recent human detections
        cursor.execute('''
            SELECT timestamp, person_id, gender, age, confidence, bbox, is_first_detection
            FROM human_detections
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        
        detections = []
        for row in cursor.fetchall():
            detections.append({
                'timestamp': row[0],
                'person_id': row[1],
                'gender': row[2],
                'age': row[3],
                'confidence': row[4],
                'bbox': row[5],
                'is_first_detection': bool(row[6])
            })
        
        conn.close()
        return jsonify(detections)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/thief_detections')
def get_thief_detections():
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Get recent thief detections
        cursor.execute('''
            SELECT timestamp, thief_name, confidence, threat_level, location, camera_id, alert_level
            FROM thief_detections
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        
        detections = []
        for row in cursor.fetchall():
            detections.append({
                'timestamp': row[0],
                'thief_name': row[1],
                'confidence': row[2],
                'threat_level': row[3],
                'location': row[4],
                'camera_id': row[5],
                'alert_level': row[6]
            })
        
        conn.close()
        return jsonify(detections)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/human_stats')
def get_human_stats():
    try:
        if security_system and security_system.human_counter:
            stats = security_system.human_counter.get_statistics()
            return jsonify(stats)
        else:
            return jsonify({'error': 'Human counter not initialized'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/recent_detections')
def get_recent_detections():
    """Get recent human detections with gender and age data"""
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Get recent detections with gender and age
        cursor.execute('''
            SELECT timestamp, person_id, gender, age, confidence, bbox, is_first_detection
            FROM human_detections
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        
        detections = []
        for row in cursor.fetchall():
            detections.append({
                'timestamp': row[0],
                'person_id': row[1],
                'gender': row[2],
                'age': row[3],
                'confidence': row[4],
                'bbox': row[5],
                'is_first_detection': bool(row[6])
            })
        
        conn.close()
        return jsonify(detections)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/gender_age_stats')
def get_gender_age_stats():
    """Get gender and age statistics"""
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Get gender distribution
        cursor.execute('''
            SELECT gender, COUNT(*) as count
            FROM human_detections
            WHERE is_first_detection = 1
            GROUP BY gender
        ''')
        gender_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get age distribution
        cursor.execute('''
            SELECT age, COUNT(*) as count
            FROM human_detections
            WHERE is_first_detection = 1
            GROUP BY age
        ''')
        age_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get total unique persons
        cursor.execute('''
            SELECT COUNT(DISTINCT person_id) as total_persons
            FROM human_detections
        ''')
        total_persons = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'gender_stats': gender_stats,
            'age_stats': age_stats,
            'total_persons': total_persons
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/humans')
def humans_page():
    return render_template('humans.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

if __name__ == '__main__':
    init_db()
    log_event("SYSTEM", "Web application started", "INFO")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
