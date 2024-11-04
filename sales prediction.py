# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv(r'C:\Users\hp\Downloads\advertising.csv')

# Display column names to confirm the correct column name
print("Columns in dataset:", data.columns)

# Data Preprocessing: Drop rows with missing values (optional)
data.dropna(inplace=True)

# Calculate Advertising Costs assuming a percentage of total spending
data['Total_Ad_Spend'] = data['TV'] + data['Radio'] + data['Newspaper']
data['Cost'] = data['Total_Ad_Spend'] * 0.6  # Assume 60% of Advertising Spend is cost

# Calculate Sales Profit and Loss
data['Profit'] = data['Sales'] - data['Cost']
total_profit = data['Profit'].sum()
total_loss = (-data[data['Profit'] < 0]['Profit']).sum()  # Only consider losses

# Prepare features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'\nMean Absolute Error (MAE) on original sales scale: {mae:.2f}')

# Plotting Actual vs Predicted Sales as a Waveform Graph
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Sales', color='blue', marker='o', linestyle='-', alpha=0.7)
plt.plot(y_pred, label='Predicted Sales', color='orange', marker='x', linestyle='--', alpha=0.7)
plt.title('Waveform of Actual vs Predicted Sales')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Graph for Total Profit and Total Loss
plt.figure(figsize=(8, 5))
plt.bar(['Total Profit', 'Total Loss'], [total_profit, total_loss], color=['green', 'red'])
plt.title('Total Profit and Loss')
plt.ylabel('Amount ($)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
