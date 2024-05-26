import tensorflow as tf
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('handwritten.h5')

# Streamlit title
st.title("Draw any number")

# Define the canvas size
canvas_width = 280  # Scale up for better drawing experience
canvas_height = 280

# Create a canvas component with thick stroke
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=10,  # Adjust this value for thicker stroke
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",
    key="canvas",
)

# Check if there's any drawing on the canvas
if canvas_result.image_data is not None:
    # Convert the canvas image data to an Image object
    img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")

    # Resize the image to 28x28 pixels
    img_28x28 = img.resize((28, 28), Image.LANCZOS)

    # Convert the image to grayscale
    img_gray = img_28x28.convert('L')

    # Invert the image colors
    img_inverted = np.invert(np.array(img_gray))

    # Ensure the values are integers in the range 0-255
    img_inverted = img_inverted.astype(np.uint8)

    # Reshape the image to match the input shape of the model
    img_input = img_inverted.reshape(1, 28, 28, 1)

    # Display the resized image
    st.image(img_28x28, caption="28x28 Pixel Image with Thick Stroke")

    # Display the grayscale image to confirm it's correctly passed
    st.image(img_gray, caption="Grayscale Image passed to variable")

    # Debugging: print shapes and some values
    st.write(f"img_inverted shape: {img_inverted.shape}")
    st.write(f"img_input shape: {img_input.shape}")
    st.write(f"img_input: {img_input[0].reshape(28, 28)}")

    # Make a prediction using the model
    prediction = model.predict(img_input)
    predicted_digit = np.argmax(prediction)

    # Display the prediction
    st.write(f"This image is probably a {predicted_digit}")

    # Optional: Display the prediction confidence
    st.write(f"Prediction confidence: {prediction[0][predicted_digit]:.2f}")
