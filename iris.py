import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target

# Data preprocessing
X = iris_data.iloc[:, :-1]  # Features
y = iris_data['species']  # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the SVM model: {accuracy:.2f}\n')

# Classification report
print("Detailed Classification Report:\n", classification_report(y_test, y_pred))

# Pair plot for visualization
sns.pairplot(iris_data, hue='species', palette='Set2', diag_kind='kde')
plt.suptitle('Iris Dataset Feature Pairplot', y=1.02, fontsize=16)
plt.show()



