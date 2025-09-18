#!/usr/bin/env python3
"""
Security System Web Application Startup Script
This script starts the Flask web application with proper error handling and setup.
"""

import sys
import os
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'opencv-python',
        'face-recognition',
        'numpy',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is available and working")
                return True
            else:
                print("⚠️  Camera is detected but not responding properly")
                return False
        else:
            print("❌ Camera is not available")
            return False
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'thief_faces',
        'known_faces',
        'templates'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    """Main startup function"""
    print("🚨 Security System Web Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check camera
    camera_available = check_camera()
    if not camera_available:
        print("⚠️  Camera not available - some features may not work")
    
    # Create directories
    create_directories()
    
    # Start the application
    print("\n🚀 Starting web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

