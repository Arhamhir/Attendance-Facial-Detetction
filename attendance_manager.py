# attendance_manager.py

import os
import pandas as pd
from datetime import datetime

# Import our custom configuration
from config import LOGS_DIR, ATTENDANCE_LOG_FILE

# Construct the full path to the log file for convenience
log_file_path = os.path.join(LOGS_DIR, ATTENDANCE_LOG_FILE)

def log_attendance(name: str):
    """
    Logs attendance for a given name into a CSV file.

    This function is idempotent for a given day: it prevents duplicate
    entries for the same person on the same calendar day.

    Args:
        name (str): The name of the person recognized.
    """
    # 1. Ensure the directory for logs exists.
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Get the current date and time
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 2. Check for duplicates for the current day.
    is_already_logged_today = False
    if os.path.exists(log_file_path):
        try:
            df = pd.read_csv(log_file_path)
            # Convert 'Timestamp' to just date for comparison
            df['Date'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d')
            
            # Check if the name exists for today's date
            if not df[(df['Name'] == name) & (df['Date'] == today_date_str)].empty:
                is_already_logged_today = True
                # print(f"Info: {name} has already been logged today.") # Optional: uncomment for debugging
        except pd.errors.EmptyDataError:
            # The file exists but is empty, so we can proceed to log.
            pass

    # 3. If not already logged, append the new entry.
    if not is_already_logged_today:
        new_log_entry = pd.DataFrame([{'Name': name, 'Timestamp': current_timestamp}])
        
        # 'mode=a' for append, 'header=False' if file exists
        # 'index=False' to not write the DataFrame index
        new_log_entry.to_csv(
            log_file_path, 
            mode='a', 
            header=not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0, 
            index=False
        )
        print(f"✅ Attendance logged for {name} at {current_timestamp}")


def get_attendance_df() -> pd.DataFrame:
    """
    Reads the attendance log CSV and returns it as a Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the attendance data.
                      Returns an empty DataFrame with correct columns if the log is empty or doesn't exist.
    """
    if not os.path.exists(log_file_path):
        # Return an empty DataFrame with the expected columns if the file doesn't exist
        return pd.DataFrame(columns=['Name', 'Timestamp'])
    
    try:
        df = pd.read_csv(log_file_path)
        # Sort by timestamp in descending order so the latest entry is on top
        df = df.sort_values(by='Timestamp', ascending=False)
        return df
    except pd.errors.EmptyDataError:
        # Return an empty DataFrame if the file is empty
        return pd.DataFrame(columns=['Name', 'Timestamp'])