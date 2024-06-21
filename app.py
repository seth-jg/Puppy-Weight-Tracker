import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file
from datetime import datetime

app = Flask(__name__)
csv_file_path = 'puppy_weights.csv'

def create_initial_csv(file_path):
    data = {
        'Name': ['Midnight', 'Trevor', 'CC', 'Lilith', 'Raven']
    }
    puppy_weight = pd.DataFrame(data)
    puppy_weight.to_csv(file_path, index=False)

def load_data(file_path):
    if not os.path.exists(file_path):
        create_initial_csv(file_path)
    return pd.read_csv(file_path)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

def add_timestamped_column(file_path, weights):
    df = load_data(file_path)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df[timestamp] = weights
    save_data(df, file_path)

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

def get_latest_column_and_names(file_path):
    df = load_data(file_path)
    latest_column = df.columns[-1] if len(df.columns) > 1 else 'No weights recorded'
    names = df['Name'].tolist()
    weights = df[latest_column].tolist() if latest_column != 'No weights recorded' else ['N/A'] * len(names)
    return list(zip(names, weights))

@app.route('/')
def index():
    latest_data = get_latest_column_and_names(csv_file_path)
    midnight = get_row_data(csv_file_path, 'Midnight')
    trevor = get_row_data(csv_file_path, 'Trevor')
    cc = get_row_data(csv_file_path, "CC")
    lilith = get_row_data(csv_file_path, "Lilith")
    raven = get_row_data(csv_file_path, 'Raven')

    return render_template('index.html', latest_data=latest_data, 
                           midnight=midnight, 
                           trevor=trevor, 
                           cc=cc, 
                           lilith=lilith, 
                           raven=raven)

@app.route('/new_weight_page', methods=["GET", "POST"])
def new_weight_page():
    latest_data = get_latest_column_and_names(csv_file_path)
    if request.method == "POST":
        weights = request.form.getlist('weights[]')
        weights = [float(weight) for weight in weights]  # Convert string inputs to floats
        add_timestamped_column(csv_file_path, weights)
        return redirect(url_for("index"))
    return render_template('update.html', latest_data=latest_data)

if __name__ == '__main__':
    app.run(debug=True)
