import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB  # Optional for text features
from sklearn.metrics import accuracy_score, classification_report
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate models based on cleaned input csv.")
parser.add_argument("base_filename", help="The base filename without extension or 'clean' (e.g., 'la_mountain_trails')")
args = parser.parse_args()

# Construct input and output filenames
input_file = args.base_filename + "-clean.csv"

# Load your cleaned DataFrame (assumes it's in a file named 'clean-la-mountain-trails-final.csv')
df = pd.read_csv(input_file)

# Split data into features (X) and target (y) 
X = df.drop('name', axis=1)  # Features (all columns except the trail name)
y = df['sac_scale']         # Target (difficulty rating)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create a dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),  # Increase max_iter 
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
    # 'Multinomial Naive Bayes': MultinomialNB()  # Uncomment if you use text features from 'name'
}

# Train each model
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained.")

# Make predictions and evaluate each model 
for name, model in models.items():
    y_pred = model.predict(X_test)
    df[f'{name} Predicted Difficulty'] = model.predict(X) # Predict for all rows
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")