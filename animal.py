import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data path
data_dir = 'raw-img'

# Image dimensions and batch size
img_height, img_width = 128, 128
batch_size = 32

# Translation dictionary
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}

# Data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(translate), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Check if a saved model exists, and if so, load it
if os.path.exists("MY_MODEL.h5"):
    model.load_weights("MY_MODEL.h5")
else:
    # Model training
    epochs = 15  # Reduced number of epochs for faster execution
    history = model.fit(
        generator,
        epochs=epochs,
        verbose=1
    )
    # Save the trained model
    model.save("MY_MODEL.h5")


# User input and prediction
def predict_animal_species(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension

    predictions = model.predict(img_array)
    class_idx = tf.argmax(predictions, axis=1)

    for name, idx in translate.items():
        if idx == class_idx:
            return name

if __name__ == "__main__":
    image_path = input("Enter the path to an animal image for species prediction: ")
    predicted_species = predict_animal_species(image_path)
    print(f"Predicted animal species: {translate[predicted_species]}")
