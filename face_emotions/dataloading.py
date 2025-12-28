from tensorflow.keras.preprocessing.image import ImageDataGenerator 

train_dir = "face_emotions/fer13/train"
test_dir = "face_emotions/fer13/test"

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

print("Class labels:", train_data.class_indices)
