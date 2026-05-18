import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

# 1. Feature Extraction (Teaching the AI what to look at)
def extract_features(img):
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG captures the "shape" of a person (head/shoulders) 
    # instead of just the color of the seat
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    hist = hog.compute(gray)
    return hist.flatten()

data = []
labels = []
dataset_path = '../data/dataset'

print("Step 1: Loading images...")
for category in ['empty', 'occupied']:
    path = os.path.join(dataset_path, category)
    label = 0 if category == 'empty' else 1
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            features = extract_features(img)
            data.append(features)
            labels.append(label)

# Convert lists to NumPy arrays for the AI
data = np.array(data)
labels = np.array(labels)

# 2. The Training Phase
# We use 20% of your data to "test" the AI after it learns
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Step 2: Training the SVM model on {len(X_train)} images...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# 3. Check the Accuracy
accuracy = model.score(X_test, y_test)
print(f"--- Training Complete! ---")
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

# 4. Save the "Trained Brain"
os.makedirs('../config', exist_ok=True)
with open('../config/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to config/model.pkl")
