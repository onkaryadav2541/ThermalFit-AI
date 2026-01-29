import tensorflow as tf
import os

# --- CONFIGURATION ---
DATASET_PATH = 'dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 8
EPOCHS = 10


def train():
    print("Loading images...")

    # 1. Load Data
    if not os.path.exists(f"{DATASET_PATH}/leak") or not os.path.exists(f"{DATASET_PATH}/no_leak"):
        print("ERROR: Please run process_video.py first to generate the images!")
        return

    # Use tf.keras.utils directly
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # 2. Build Model (Using full paths to fix red lines)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)  # Output: [No_Leak, Leak]
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 3. Train
    print("Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 4. Save
    model.save('mask_fit_model.h5')
    print("SUCCESS! Model saved.")


if __name__ == "__main__":
    train()