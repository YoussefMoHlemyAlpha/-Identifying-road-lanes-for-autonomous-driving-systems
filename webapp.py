from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import base64
import traceback

app = FastAPI()   # Creates a FastAPI instance

# Load the model (ensure the path is correct)
try:
    model = tf.keras.models.load_model('D:/imageproject/unet_model.h5')
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    model = None

@app.get("/", response_class=HTMLResponse)
async def get_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Prediction</title>
    </head>
    <body>
        <h1>Image Prediction System</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label for="file">Choose an image please :</label><br><br>
            <input type="file" name="file" id="file" required><br><br>
            <button type="submit">Predict</button>
        </form>
        <br>
        <div id="prediction-result">
            <h2>Prediction Result:</h2>
            <img id="predicted-image" src="" alt="Predicted Image" style="max-width: 100%;"/>
        </div>

        <script>
            
            const form = document.querySelector("form");

            <!-- Event Listener -->
            form.addEventListener("submit", async function (event) {  
                event.preventDefault();
                const formData = new FormData(form);

            <!--Sending the Request to the Server-->
                const response = await fetch("/predict", { 
                    method: "POST",
                    body: formData,
                });
                
                const result = await response.json();
                <!--if the server's response contains a key prediction_image -->
                if (result.prediction_image) {
                    document.getElementById("predicted-image").src = "data:image/png;base64," + result.prediction_image;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
  # request to the /predict endpoint and displaying the predicted image result.
@app.post("/predict", response_class=JSONResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded correctly.")

    try:
        # Read the image file and open it
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image (resize to 80x80, convert to grayscale, and normalize)
        image = image.resize((80, 80))  # Resize to the expected size
        image = image.convert('L')  # Convert to grayscale (1 channel)
        image = np.array(image)  # Convert to numpy array
        image = image / 255.0  # Normalize

        # Add batch dimension for prediction
        image = np.expand_dims(image, axis=0)  # Shape (1, 80, 80, 1)

        # Log the shape of the input image before prediction
        print(f"Input image shape: {image.shape}")

        # Make prediction
        pred = model.predict(image)

        # Log the prediction result
        print(f"Prediction result: {pred}")

        # If the model is a segmentation model, threshold the output to create a binary mask
        pred_mask = (pred > 0.5).astype(np.uint8)  # Convert to binary mask (0 or 1)

        # Convert the prediction to an image format (grayscale image for segmentation)
        pred_img = Image.fromarray(pred_mask[0, :, :, 0] * 255) # Convert to 0-255 range

        # Create an in-memory byte buffer to save the image data temporarily
        buffered = io.BytesIO()
        # Save the image data in PNG format into the in-memory buffer
        pred_img.save(buffered, format="PNG")
        # Encode the binary image data from the buffer into a Base64 string for easier transmission
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the predicted image as a base64 string
        return JSONResponse(content={"prediction_image": encoded_image})

    except Exception as e:
        # Log the error and return a detailed message
        error_message = f"Error during prediction: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_message)
