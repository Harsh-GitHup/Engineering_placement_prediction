import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Specify the absolute path to the CSV file
file_path = r"E:\Projects\IBM Hack Chall. Project\Dataset\Placement_Data_Full_Class.csv"

# Step 1: Import and Load Data
data = pd.read_csv(file_path)

# Print out column names to verify the correct names in your dataset
print(data.columns)

# Step 2: Preprocess Data
# For simplicity, let's handle missing values by dropping them and encoding categorical variables using one-hot encoding.
data = data.dropna()  # Drop rows with missing values
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})  # Convert target variable to binary values

data = pd.get_dummies(data, columns=['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation'], drop_first=True)

# Step 3: Explore and Visualize Data
# Use various visualization techniques (histograms, scatter plots, etc.) to understand the data distribution and relationships.

# Step 4: Choose a Machine Learning Algorithm
X = data.drop('status', axis=1)  # Features
Y = data['status']  # Target variable

# Check unique classes in the target variable
unique_classes = np.unique(Y)
print("Unique classes in y:", unique_classes)


# Step 5: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 6: Train a Machine Learning Model
model = LogisticRegression(solver='liblinear')  # You can replace this with any other suitable algorithm
model.fit(X_train, Y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

# Additional Step: Visualize Results
sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()