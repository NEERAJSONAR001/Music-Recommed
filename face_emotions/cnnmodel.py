import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Flatten, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DIR = "face_emotions/fer13/train"
TEST_DIR  = "face_emotions/fer13/test"

datagen = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=10,         
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("Emotion Classes:", train_data.class_indices)


model = Sequential()

# Block 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))

# Block 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Block 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_data.num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

test_loss, test_accuracy = model.evaluate(test_data)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

# DETAILED METRICS
y_true = test_data.classes
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=test_data.class_indices.keys()
))

model.save("face_emotions/face_emotion_cnn_model.h5")
print("\nModel saved successfully.")
