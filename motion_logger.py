import csv
from datetime import datetime

def log_motion(avg_motion, motion_level, events, frames_analyzed, extra_info=''):
    file = 'motion_log.csv'
    header = ['timestamp', 'avg_motion', 'motion_level', 'events', 'frames_analyzed', 'extra_info']
    row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), avg_motion, motion_level, events, frames_analyzed, extra_info]
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
        print(f'Error logging motion: {e}') 