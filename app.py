import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
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
    return df.iloc[name]

def get_latest_column_and_names(file_path):
    df = load_data(file_path)
    latest_column = df.columns[-1]
    names = df['Name'].tolist()
    weights = df[latest_column].tolist()
    return list(zip(names, weights))

#create_initial_csv(csv_file_path)
#weights = [10, 12, 8, 15, 7]  # Example weights
#add_timestamped_column(csv_file_path, weights)


@app.route('/')
def index():
    latest_data = get_latest_column_and_names(csv_file_path)
    return render_template('index.html', latest_data=latest_data)

@app.route('/new_weight_page', methods=["GET", "POST"])
def new_weight_page():
    latest_data = get_latest_column_and_names(csv_file_path)
    if request.method == "POST":
        weights = request.form.getlist('weights[]')
        add_timestamped_column(csv_file_path, weights)
        return redirect(url_for("index"))
    return render_template('update.html', latest_data=latest_data)

if __name__ == '__main__':
    app.run(debug=True)