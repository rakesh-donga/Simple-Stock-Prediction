import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_excel('TATA.xlsx')
print("Initial DataFrame:")
print(df.head())

numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in numeric_columns:
    df[col] = df[col].interpolate(method='linear')

def date_to_excel_serial(date):
    base_date = datetime(1899, 12, 30)
    return (date - base_date).days

df['Date_Serial'] = df['Date'].apply(date_to_excel_serial)

X = df[['Date_Serial']].values
y = df['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Performance (Mean Absolute Error):")
print(np.mean(np.abs(y_test - y_pred)))

def user_date_to_excel_serial(date_str):
    try:
        input_date = datetime.strptime(date_str, '%Y-%m-%d')
        base_date = datetime(1899, 12, 30)
        return (input_date - base_date).days
    except ValueError:
        return None

def excel_serial_to_date(serial):
    base_date = datetime(1899, 12, 30)
    return base_date + pd.to_timedelta(serial, unit='D')

input_date = input("\nEnter the date for prediction (YYYY-MM-DD, e.g., 2025-09-28): ")
serial_date = user_date_to_excel_serial(input_date)

if serial_date is not None:
    print(f"Converted input date to Excel serial date: {serial_date}")
    predicted_close = model.predict([[serial_date]])
    print(f"Predicted Close Price for {input_date}: {predicted_close[0]:.2f}")

    df['Date_Converted'] = df['Date_Serial'].apply(excel_serial_to_date)
    predicted_date = excel_serial_to_date(serial_date)

    plt.figure(figsize=(10, 6))
    plt.plot(df['Date_Converted'], df['Close'], label='Historical Close Price', color='blue')
    plt.scatter([predicted_date], predicted_close, color='red', label='Predicted Close', s=100)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('TATA Stock Close Price with Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Invalid date format. Please use YYYY-MM-DD (e.g., 2025-09-28).")
