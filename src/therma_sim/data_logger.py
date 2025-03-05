import csv
import os

class DataLogger:
    def __init__(self):
        """Initialize a flexible data logger that writes directly to CSV files."""
        self.files = {}

    def make_data_reporter(self, file_name, column_names):
        """
        Create a new CSV file and overwrite any existing file with the same name.

        file_name: Name of the CSV file (string, e.g., "agents.csv").
        column_names: List of column names (e.g., ["Step", "AgentID", "Energy", "BodyTemp"]).
        """
        # Always overwrite the file
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)  # Write column headers

        # Store reference for tracking
        self.files[file_name] = column_names  

    def log_data(self, file_name, data):
        """
        Write a row of raw data directly to CSV.

        file_name: CSV file name (string, e.g., "agents.csv").
        data_row: List of values (must match column order).
        """
        if file_name not in self.files:
            raise ValueError(f"File '{file_name}' has not been initialized with make_data_reporter.")

        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
