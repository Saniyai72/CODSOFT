import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
movie_data = pd.read_csv("C:\\Users\\hp\\Downloads\\documents\\IMDb Movies India.csv", encoding='ISO-8859-1')

# Data preprocessing
movie_data.dropna(subset=['Rating', 'Genre', 'Director', 'Votes'], inplace=True)

# Feature engineering
X = movie_data[['Genre', 'Director', 'Votes']]
y = movie_data['Rating'].dropna()

# Convert categorical features using one-hot encoding
X = pd.get_dummies(X, columns=['Genre', 'Director'], drop_first=True)

# Clean 'Votes' column by removing non-numeric characters and converting to float
X['Votes'] = X['Votes'].str.replace(r'[\$,M]', '', regex=True).astype(float)

# Align X and y after dropping NaN values
X = X.loc[y.index]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot Actual vs Predicted Ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predicted Ratings')
plt.scatter(y_test, y_test, alpha=0.6, color='blue', label='Actual Ratings')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit Line')
plt.title('Actual vs Predicted Ratings', fontsize=16)
plt.xlabel('Actual Ratings', fontsize=14)
plt.ylabel('Predicted Ratings', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('actual_vs_predicted_ratings.png')
plt.show()