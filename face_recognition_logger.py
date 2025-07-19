import csv
from datetime import datetime

def log_face_recognition(event_type, name, confidence, person_type, image_path, extra_info=''):
    file = 'face_recognition_log.csv'
    header = ['timestamp', 'event_type', 'name', 'confidence', 'person_type', 'image_path', 'extra_info']
    row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), event_type, name, confidence, person_type, image_path, extra_info]
    try:
        write_header = False
        try:
            with open(file, 'r') as f:
                if not f.readline():
                    write_header = True
        except FileNotFoundError:
            write_header = True
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(f'Error logging face recognition: {e}') 