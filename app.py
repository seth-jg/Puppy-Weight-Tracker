# Import necessary modules
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)
csv_file_path = 'puppy_weights.csv'

# Function to create initial CSV if not exists
def create_initial_csv(file_path):
    data = {
        'Name': ['Midnight', 'Trevor', 'CC', 'Lilith', 'Raven']
    }
    puppy_weight = pd.DataFrame(data)
    puppy_weight.to_csv(file_path, index=False)

# Function to load data from CSV
def load_data(file_path):
    if not os.path.exists(file_path):
        create_initial_csv(file_path)
    return pd.read_csv(file_path)

# Function to save data to CSV
def save_data(df, file_path):
    df.to_csv(file_path, index=False)

# Function to add timestamped column with weights
def add_timestamped_column(file_path, weights):
    df = load_data(file_path)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df[timestamp] = weights
    save_data(df, file_path)

# Function to get row data (weights history) for a puppy
def get_row_data(file_path, name):
    df = load_data(file_path)
    if name in df['Name'].values:
        weights = []
        dates = []
        for column in df.columns[1:]:  # Start from the second column, assuming first column is 'Name'
            weight = df[df['Name'] == name][column].values[0]
            if not pd.isnull(weight):
                weights.append(weight)
                dates.append(column)
        return list(zip(dates, weights))
    else:
        return None

# Function to get latest column and names from CSV
def get_latest_column_and_names(file_path):
    df = load_data(file_path)
    latest_column = df.columns[-1] if len(df.columns) > 1 else 'No weights recorded'
    names = df['Name'].tolist()
    weights = df[latest_column].tolist() if latest_column != 'No weights recorded' else ['N/A'] * len(names)
    return list(zip(names, weights)), list(df.columns[1:])  # Also return list of dates

# Route for the home page
@app.route('/')
def index():
    date = request.args.get('date')
    df = load_data(csv_file_path)
    
    dates = list(df.columns[1:])  # Get all dates
    if date and date in dates:
        weights_data = list(zip(df['Name'], df[date].fillna('N/A')))
    else:
        weights_data, dates = get_latest_column_and_names(csv_file_path)
        date = dates[-1] if dates else None

    midnight = get_row_data(csv_file_path, 'Midnight')
    trevor = get_row_data(csv_file_path, 'Trevor')
    cc = get_row_data(csv_file_path, "CC")
    lilith = get_row_data(csv_file_path, "Lilith")
    raven = get_row_data(csv_file_path, 'Raven')

    return render_template('index.html', latest_data=weights_data, 
                           midnight=midnight, 
                           trevor=trevor, 
                           cc=cc, 
                           lilith=lilith, 
                           raven=raven, 
                           dates=dates,
                           names=df['Name'],
                           selected_date=date)


# Route for adding new weight page
@app.route('/new_weight_page', methods=["GET", "POST"])
def new_weight_page():
    latest_data, _ = get_latest_column_and_names(csv_file_path)
    if request.method == "POST":
        weights = request.form.getlist('weights[]')
        weights = [float(weight) for weight in weights]  # Convert string inputs to floats
        add_timestamped_column(csv_file_path, weights)
        return redirect(url_for("index"))
    return render_template('update.html', latest_data=latest_data)

# Route for comparing weights
@app.route('/compare_weights')
def compare_weights():
    puppy1 = request.args.get('puppy1')
    puppy2 = request.args.get('puppy2')
    
    data_puppy1 = get_row_data(csv_file_path, puppy1)
    data_puppy2 = get_row_data(csv_file_path, puppy2)
    
    # Ensure both puppies have data
    if data_puppy1 and data_puppy2:
        # Create a list of dictionaries containing the dates and weights for both puppies
        comparison_data = []
        for (date1, weight1), (date2, weight2) in zip(data_puppy1, data_puppy2):
            if date1 == date2:
                comparison_data.append({
                    'date': date1,
                    'weight1': weight1,
                    'weight2': weight2
                })
        
        return jsonify(comparison_data)
    else:
        return jsonify([])  # Return empty list if no data found for any puppy

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)