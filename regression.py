import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def polynomial_regression_forecast(filename, degree=3):
    # Load the dataset
    data = pd.read_csv(filename)

    # Extract the dates
    dates = data.columns[1:]

    # Convert the dates to a datetime format
    dates = pd.to_datetime(dates)

    # Iterate over each row in the dataset
    for index, row in data.iterrows():
        puppy_name = row['Name']
        weights = row[1:].values.astype(float)

        # Prepare the data
        df = pd.DataFrame({'date': dates, 'weight': weights})
        df.set_index('date', inplace=True)
        df['day_number'] = np.arange(len(df))
        X = df['day_number'].values.reshape(-1, 1)
        y = df['weight'].values

        # Create a polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)

        # Predict for the next 7 days
        last_day = df['day_number'].iloc[-1]
        future_days = np.arange(last_day + 1, last_day + 8).reshape(-1, 1)
        future_predictions = model.predict(future_days)

        # Create future dates
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

        # Plot the data and the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['weight'], label='Actual Data', marker='o')
        plt.plot(future_dates, future_predictions, label='Predicted Data', marker='x', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.title(f'Weight Prediction for {puppy_name} for the Next 7 Days')
        plt.legend()
        plt.grid(True)
        
        # Save the plot as a JPG file
        plt.savefig(f'{puppy_name}_weight_prediction.jpg')
        plt.close()


# Example usage:
# Assuming the CSV file is named 'puppy_weight.csv'
polynomial_regression_forecast('puppy_weights.csv')