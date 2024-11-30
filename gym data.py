# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv(r"C:\Users\Sudharsan\Downloads\archive\gym_members_exercise_tracking.csv")

# Define features and target for Linear Regression
features = ['BMI', 'Session_Duration (hours)', 'Max_BPM', 'Age']
target = 'Calories_Burned'

X = data[features]
y = data[target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression R² Score: {r2_lr}")

# Linear Regression Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.7, color="blue")
plt.axline((0, 0), slope=1, color="red", linestyle="--", linewidth=1.5)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.show()

# Preparing data for KNN
knn_features = ['BMI', 'Session_Duration (hours)', 'Max_BPM', 'Age']
knn_target = 'Workout_Type'  # Assuming Workout_Type is a classification column in your dataset

# Encode target if categorical
if data[knn_target].dtype == 'object':
    data[knn_target] = data[knn_target].astype('category').cat.codes

X_knn = data[knn_features]
y_knn = data[knn_target]

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42)

# Optimal k Search
accuracies = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn.predict(X_test_knn)
    accuracies.append(accuracy_score(y_test_knn, y_pred_knn))

# Plot Optimal k Search
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title("KNN: Optimal k Search")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()

# Train KNN with optimal k
optimal_k = k_values[np.argmax(accuracies)]
knn_best = KNeighborsClassifier(n_neighbors=optimal_k)
knn_best.fit(X_train_knn, y_train_knn)
y_pred_best_knn = knn_best.predict(X_test_knn)

# KNN Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_knn['BMI'], y=y_test_knn, label="Actual", color="blue", alpha=0.6)
sns.scatterplot(x=X_test_knn['BMI'], y=y_pred_best_knn, label="Predicted", color="orange", alpha=0.6)
plt.title(f"KNN Scatter Plot (k={optimal_k})")
plt.xlabel("BMI")
plt.ylabel("Workout Type")
plt.legend()
plt.show()

