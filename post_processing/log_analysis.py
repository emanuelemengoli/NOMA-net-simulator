import pandas as pd
import json

def load_simulation_data(filepath):
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file if line.strip()]

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Path to the JSON log file
    log_filepath = '/output/simulation_logs.json'
    
    # Load the data into a DataFrame
    simulation_df = load_simulation_data(log_filepath)
    
    # Display the DataFrame
    print(simulation_df)
