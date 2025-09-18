#!/usr/bin/env python3
"""
Test script for gender and age detection in the web application
"""

import sys
import os
import sqlite3
import csv

def test_database_setup():
    """Test if the database is properly set up"""
    print("🧪 Testing Database Setup")
    print("=" * 30)
    
    try:
        conn = sqlite3.connect('security_system.db')
        cursor = conn.cursor()
        
        # Check if human_detections table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='human_detections'")
        if cursor.fetchone():
            print("✅ human_detections table exists")
        else:
            print("❌ human_detections table not found")
            return False
        
        # Check table structure
        cursor.execute("PRAGMA table_info(human_detections)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        required_columns = ['timestamp', 'person_id', 'gender', 'age', 'confidence', 'bbox', 'is_first_detection']
        for col in required_columns:
            if col in column_names:
                print(f"✅ Column '{col}' exists")
            else:
                print(f"❌ Column '{col}' missing")
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_csv_import():
    """Test importing data from CSV file"""
    print("\n📊 Testing CSV Import")
    print("=" * 25)
    
    try:
        if not os.path.exists('human_detection_data.csv'):
            print("❌ CSV file not found")
            return False
        
        # Read CSV file
        with open('human_detection_data.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"✅ Found {len(rows)} records in CSV file")
        
        # Check if data has gender and age
        if rows:
            first_row = rows[0]
            if 'gender' in first_row and 'age' in first_row:
                print(f"✅ Gender data: {first_row['gender']}")
                print(f"✅ Age data: {first_row['age']}")
                return True
            else:
                print("❌ Gender or age data missing from CSV")
                return False
        else:
            print("❌ No data in CSV file")
            return False
            
    except Exception as e:
        print(f"❌ CSV import error: {e}")
        return False

def test_web_app_apis():
    """Test if web app APIs work"""
    print("\n🌐 Testing Web App APIs")
    print("=" * 30)
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test gender/age stats API
            response = client.get('/api/gender_age_stats')
            if response.status_code == 200:
                data = response.get_json()
                print("✅ Gender/age stats API working")
                print(f"   Gender stats: {data.get('gender_stats', {})}")
                print(f"   Age stats: {data.get('age_stats', {})}")
            else:
                print(f"❌ Gender/age stats API failed: {response.status_code}")
                return False
            
            # Test recent detections API
            response = client.get('/api/recent_detections')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Recent detections API working ({len(data)} detections)")
            else:
                print(f"❌ Recent detections API failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Web app API error: {e}")
        return False

def main():
    """Main test function"""
    print("🚨 Gender & Age Detection Test")
    print("=" * 40)
    
    # Test database
    db_ok = test_database_setup()
    
    # Test CSV import
    csv_ok = test_csv_import()
    
    # Test web app APIs
    api_ok = test_web_app_apis()
    
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Database Setup: {'✅ PASS' if db_ok else '❌ FAIL'}")
    print(f"CSV Import: {'✅ PASS' if csv_ok else '❌ FAIL'}")
    print(f"Web App APIs: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if db_ok and csv_ok and api_ok:
        print("\n🎉 All tests passed! Gender and age detection should work in the web app.")
        print("   Run: python app.py")
        print("   Then check the dashboard for gender/age statistics")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

