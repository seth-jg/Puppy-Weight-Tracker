import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

def regression(file_path, name, plot_path, days_ahead=5, max_weight=10000):
    dataset = get_row_data(file_path, name)
    if dataset is None:
        print(f"No data found for '{name}'")
        return
    
    # Extract dates and weights from dataset
    dates, weights = zip(*dataset)
    
    # Convert dates to numerical representation (for simplicity, converting to indices)
    X = np.arange(len(dates)).reshape(-1, 1)  # Reshape to 2D array (n_samples, n_features)
    y = np.array(weights)                     # Convert weights to numpy array
    
    # Fit polynomial regression
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    # Predict weights for future days
    future_X = np.arange(len(dates), len(dates) + days_ahead).reshape(-1, 1)
    future_X_poly = poly_reg.transform(future_X)
    future_y_pred = lin_reg.predict(future_X_poly)

    # Plotting
    X_grid = np.arange(min(X), max(future_X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.figure()
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue', label='Polynomial Regression')
    
    # Plot future predictions
    plt.plot(future_X, future_y_pred, color='green', linestyle='--', label=f'Predicted for {days_ahead} days ahead')
    
    plt.title(f'Puppy Weight Prediction for {name}')
    plt.xlabel('Days')
    plt.ylabel('Weight')
    plt.legend()
    
    # Set maximum weight for the y-axis if specified
    if max_weight is not None:
        plt.ylim(0, max_weight)
    
    # Save plot as an image
    plt.savefig(plot_path)
    plt.close()

    # Return regression model or other outputs as needed
    return lin_reg


@app.route('/')
def index():
    latest_data = get_latest_column_and_names(csv_file_path)
    midnight = get_row_data(csv_file_path, 'Midnight')
    trevor = get_row_data(csv_file_path, 'Trevor')
    cc = get_row_data(csv_file_path, "CC")
    lilith = get_row_data(csv_file_path, "Lilith")
    raven = get_row_data(csv_file_path, 'Raven')

    latest_data = get_latest_column_and_names(csv_file_path)
    midnight_plot_url = url_for('static', filename='img/midnight_plot.png')
    trevor_plot_url = url_for('static', filename='img/trevor_plot.png')
    cc_plot_url = url_for('static', filename='img/cc_plot.png')
    lilith_plot_url = url_for('static', filename='img/lilith_plot.png')
    raven_plot_url = url_for('static', filename='img/raven_plot.png')

    return render_template('index.html', latest_data=latest_data, 
                           midnight=midnight, 
                           trevor=trevor, 
                           cc=cc, 
                           lilith=lilith, 
                           raven=raven, 
                           midnight_plot_url=midnight_plot_url,
                           trevor_plot_url=trevor_plot_url,
                           cc_plot_url=cc_plot_url,
                           lilith_plot_url=lilith_plot_url,
                           raven_plot_url=raven_plot_url)

@app.route('/new_weight_page', methods=["GET", "POST"])
def new_weight_page():
    latest_data = get_latest_column_and_names(csv_file_path)
    if request.method == "POST":
        weights = request.form.getlist('weights[]')
        weights = [float(weight) for weight in weights]  # Convert string inputs to floats
        add_timestamped_column(csv_file_path, weights)
        regression(csv_file_path, "Midnight", 'static/img/midnight_plot.png')
        regression(csv_file_path, "Trevor", 'static/img/trevor_plot.png')
        regression(csv_file_path, "CC", 'static/img/cc_plot.png')
        regression(csv_file_path, "Lilith", 'static/img/lilith_plot.png')
        regression(csv_file_path, "Raven", 'static/img/raven_plot.png')

        return redirect(url_for("index"))
    return render_template('update.html', latest_data=latest_data)

if __name__ == '__main__':
    app.run(debug=True)
