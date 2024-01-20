import sys
import os
import pandas as pd


def load_csv():
    crypto_data_objects = []
    # Check if a file path is provided
    if len(sys.argv) < 2:
        print("Usage: python app.py <path_to_csv_file>")
        sys.exit(1)

    # Read CSV file path from command line
    csv_directory_path = sys.argv[1]

    # Load data
    try:
        csv_files = [
            os.path.join(csv_directory_path, file)
            for file in os.listdir(csv_directory_path)
            if file.endswith(".csv")
        ]
        for f in csv_files:
            crypto_name = os.path.basename(f).split("_")[0]
            df = pd.read_csv(f)
            crypto_data_objects.append({"crypto": crypto_name, "dataFrame": df})
        return crypto_data_objects
    except Exception as e:
        print(f"Failed to read the CSV file: {e}")
        sys.exit(1)
