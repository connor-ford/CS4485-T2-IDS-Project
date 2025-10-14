import json
from datetime import datetime
import os

# Path relative to app.py location
JSON_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.json")

def save_result_to_json(new_data):
    # Ensure timestamp added
    new_data["timestamp"] = datetime.utcnow().isoformat()

    # Load existing database or initialize empty
    if os.path.exists(JSON_DB_PATH):
        with open(JSON_DB_PATH, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append and write back
    data.append(new_data)
    with open(JSON_DB_PATH, "w") as file:
        json.dump(data, file, indent=4)
