import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = np.loadtxt('./AI-teach/ai_train_masterclass/data.txt')
X = data[:, :-1]
Y = data[:, -1]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)