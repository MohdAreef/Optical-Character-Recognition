import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import cv2
# import keras
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer


# Title of the app
st.title('OCR ')
# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_model():
    try:
        # model = tf.keras.models.load_model("C:/Users/MOHD AREEF/Downloads/model2.keras")
        model = tf.keras.models.load_model("./mynewmodel.keras")
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.write(f"Error loading model: {e}")
        st.error("Error loading model.")   
        return None
@st.cache_resource
def load_resnet():
    try:
        model = tf.keras.models.load_model("./resnetmodel1.keras") 
        st.write("Resnet Model loaded successfully.")
        return model
    except Exception as e:
        st.write(f"Error loading model: {e}")
        st.error("Error loading model.")
        return None
@st.cache_resource   
def load_tokenizer():
    try:
        # model = tf.keras.models.load_model("C:\Users\MOHD AREEF\Downloads\tokenizer.pickle")
        # st.write("Resnet Model loaded successfully.")
        # Load the tokenizer from the file
        with open('./tokenizer.pickle', 'rb') as handle:
           tokenizer = pickle.load(handle)
        st.write("Model loaded successfully.")
        return tokenizer
    except Exception as e:
        st.write(f"Error loading model: {e}")
        st.error("Error loading model.")
        return None
    
model = load_model()
pretrained_cnn = load_resnet()
tokenizer = load_tokenizer()

# Function to preprocess the image
def preprocess_image(image, target_size=(128, 128)):
    target_height, target_width = target_size 

      # Get original dimensions
    original_height, original_width = image.shape

 



    # Calculate the scaling factor and resize dimensions
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image with maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height)) 

    # Create a new image with the target size and fill with white (255)
    padded_image = np.full((target_height, target_width), 255, dtype=np.uint8)

    # Calculate padding
    pad_top = (target_height - new_height) // 2
    pad_left = (target_width - new_width) // 2

    # Place the resized image in the center
    padded_image[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized_image

    # Normalize the image (convert pixel values to range [0, 1])
    normalized_image = padded_image / 255.0
    
    X_train=np.array(normalized_image)

    # Expand dimensions to add a single channel
    X_train_expanded = np.expand_dims(X_train, axis=-1)

# Replicate the single channel 3 times to create an RGB-like image
    X_train_rgb = np.repeat(X_train_expanded, 3, axis=-1)

    # print("x_train_rgb return image=",X_train_rgb)
    return X_train_rgb

#def fun to extract features
def extractfeatures(X_train_rgb):

    print("********************************X_train_rgb IMAGE SIZE*************************",X_train_rgb.shape)
    # Extract features from preprocessed images using MobileNetV2
    features_train = pretrained_cnn.predict(X_train_rgb)
    print("********************************X_train_rgb IMAGE SIZE*************************",X_train_rgb.shape)
    # Flatten the feature vectors
    features_train_flattened = features_train.reshape(features_train.shape[0], -1) 
    print("feature_train_flatten shape",features_train.shape)
     
    return features_train_flattened




# Function to perform OCR using the custom model
def ocr_predict(image):
    # image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image=preprocess_image(image) 
    # image = np.array(image)
    
    image = image.reshape(1, 128, 128, 3)
    print("********************************FEATURE TRAIN FLATTEN pre_image*************************",image.shape)
    features_train_flattened=extractfeatures(image)
    print("********************************FEATURE TRAIN FLATTEN IMAGE SIZE*************************",features_train_flattened.shape)
    prediction=model.predict(features_train_flattened)
    
    # return "hello"
    return prediction

# Function to decode model prediction to text
# This function needs to be implemented according to your model's output
def decode_prediction(prediction,tokenizer):
    decoded_words = []
    print("DICTINARY",tokenizer.index_word)
    for pred in prediction:
        # Convert each time step's probabilities into character indices
        character_indices = [np.argmax(step) for step in pred]
        print("INDICES=",character_indices)
#         print(character_indices)
        # Decode character indices into characters using the tokenizer
        characters = [tokenizer.index_word.get(idx, '') for idx in character_indices]
        print("CHARACTERS",characters)
#         print(characters)
        # Combine characters into words based on spaces (assuming space-separated words)
        word = ''.join(characters).strip()  # Combine characters into a word
        decoded_words.append(word)
    # st.write(tokenizer.index_word)   
    return decoded_words   
    # pass
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:

     # Step 1: Read the file content
    file_bytes = uploaded_file.read()

    # Step 2: Convert the bytes data to a NumPy array
    np_array = np.frombuffer(file_bytes, np.uint8)

    # Step 3: Decode the NumPy array to an image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Optional: Convert the image to grayscale if needed
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

     # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform OCR and get the result
    result_text = ocr_predict(image) 
    # st.write(result_text)
    result=decode_prediction(result_text,tokenizer) 
    # st.write(tokenizer)
    # Display the result
    st.text_area("OCR Result", result[0], height=200)
else:
        # The image is already in grayscale
    st.write("Please upload an image.")

              
   

# # A submit button (though the function executes upon file upload, this is optional)
# if st.button("Submit"):
#     if uploaded_file is not None:
#         result_text = ocr_predict(image)
#         st.text_area("OCR Result", result_text, height=200)
#     else:
#         st.write("Please upload an image first.")
