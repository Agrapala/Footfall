import csv
from datetime import datetime

def log_unique_count(method, unique_count, window_minutes, extra_info=''):
    file = 'unique_count_log.csv'
    header = ['timestamp', 'method', 'unique_count', 'window_minutes', 'extra_info']
    row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), method, unique_count, window_minutes, extra_info]
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
        print(f'Error logging unique count: {e}') 