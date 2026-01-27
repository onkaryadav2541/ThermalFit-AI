import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- SETTINGS ---
# Ensure this matches your folder name exactly
DATASET_PATH = r'dataset_train'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 4  # Small batch size for 48 images
EPOCHS = 20  # Number of training loops


# ----------------

def train_model():
    # 1. SETUP DATA AUGMENTATION
    # This creates "fake" copies of your 48 images (rotates, zooms)
    # so the model has more data to learn from.
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Uses 20% of images to test itself
    )

    # 2. LOAD IMAGES FROM YOUR FOLDERS
    print("Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # This detects your folder names (downleak, straightleak) automatically
    class_names = list(train_generator.class_indices.keys())
    print(f"✅ FOUND CLASSES: {class_names}")

    # 3. BUILD THE BRAIN (CNN Architecture)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. START TRAINING
    print("🚀 Starting Training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # 5. SAVE THE RESULT
    model.save('mask_leak_model.h5')

    # Save the class names to a text file for later use
    with open('class_names.txt', 'w') as f:
        f.write(','.join(class_names))

    print("\n------------------------------------------------")
    print("🎉 TRAINING FINISHED!")
    print("Model saved as: mask_leak_model.h5")
    print("Class names saved as: class_names.txt")
    print("------------------------------------------------")


if __name__ == '__main__':
    train_model()