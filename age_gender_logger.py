import csv
from datetime import datetime

def log_age_gender(face_id, age_range, gender, confidence, bbox, extra_info=''):
    file = 'age_gender_log.csv'
    header = ['timestamp', 'face_id', 'age_range', 'gender', 'confidence', 'bbox', 'extra_info']
    row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), face_id, age_range, gender, confidence, bbox, extra_info]
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
        print(f'Error logging age/gender: {e}') 