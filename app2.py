import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from translate import translate  


model = keras.models.load_model('MY_MODEL.h5')

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  
    img = img / 255.0  
    return np.expand_dims(img, axis=0)

# Function to predict animal species
def predict_animal_species(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_idx = np.argmax(prediction)
    
    # Convert predicted_class_idx to the corresponding Italian animal species string
    predicted_species_italian = list(translate.keys())[predicted_class_idx]
    
    # Translate Italian to English
    predicted_species_english = translate.get(predicted_species_italian, "Translation Not Found")
    
    return predicted_species_english

if __name__ == "__main__":
    while True:
        image_path = input("Enter the path to an image for animal species prediction (or 'exit' to quit): ")
        
        if image_path.lower() == 'exit':
            break
        
        try:
            predicted_species = predict_animal_species(image_path)
            print(f"Predicted animal species (English): {predicted_species}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
