import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_preprocess

# Load data
X, y, scaler, le = load_and_preprocess("data/merged_exoplanets.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report only if label encoder has classes
if hasattr(le, 'classes_'):
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and scaler
with open('models/exo_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'label_encoder': le}, f)

print("Model saved successfully!")
