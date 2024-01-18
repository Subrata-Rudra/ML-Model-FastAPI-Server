from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("potato_model_2.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/test")
async def test():
    return "Server is OK✅"


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image


def process_image(file_content):
    image = Image.open(BytesIO(file_content))
    image = image.convert("RGB").resize((256, 256))
    return np.array(image)


@app.post("/predict")
async def predict(image_to_predict: UploadFile = File(...)):
    # image = read_file_as_image(await image_to_predict.read())
    # image_batch = np.expand_dims(image, 0)
    # prediction = MODEL.predict(image_batch)
    # predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    # confidence = round(np.max(prediction[0]) * 100, 2)
    # return {"Class": predicted_class, "Confidence": confidence}
    image_content = await image_to_predict.read()
    processed_image = process_image(image_content)
    image_batch = np.expand_dims(processed_image, 0)
    prediction = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = round(np.max(prediction[0]) * 100, 2)
    return {"Class": predicted_class, "Confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
