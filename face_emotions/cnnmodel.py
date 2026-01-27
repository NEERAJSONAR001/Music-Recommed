import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,classification_report

EMOTION_LABELS = [
    "surprise",  
    "happy",    
    "sad",       
    "angry",     
    "neutral"    
]


TRAIN_DIR = "face_emotions/raf/DATASET/train"
TEST_DIR  = "face_emotions/raf/DATASET/test"

IMG_SIZE = (96, 96)
BATCH_SIZE = 32


train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

print("Train Class labels:", train_data.class_indices)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes


base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(96, 96, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🔵 Stage 1: Training classifier head\n")
model.fit(train_data, epochs=15, validation_data=test_data)


for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🟢 Stage 2: Fine-tuning top layers\n")
model.fit(train_data, epochs=30, validation_data=test_data)


loss, acc = model.evaluate(test_data)
print(f"\n🔥 FINAL ACCURACY: {acc * 100:.2f}%")

y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


model.save("face_emotions/face_emotion_mobilenetv2_70.h5")
print("\nModel saved successfully.")
