import cv2
import numpy as np
import logging

from fastapi import FastAPI, File, status, UploadFile
from fastapi.responses import JSONResponse

from obstacle_detection import process_frame
from data import ErrorResponse, SuccessResponse

app = FastAPI()

logging.basicConfig(level=logging.INFO)


@app.post("/process-image/", response_model=SuccessResponse, status_code=status.HTTP_200_OK)
async def process_image_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            logging.error("Error: Uploaded file is empty.")
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=ErrorResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                message="Uploaded file is empty",
                error="No content in the uploaded file."
            ).model_dump())

        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            logging.error("Error: Could not read the image. Make sure the uploaded file is a valid image.")
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=ErrorResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                message="Invalid image file",
                error="Could not decode the image."
            ).model_dump())

        objects_distances_list = process_frame(image)

        return SuccessResponse(
            status_code=status.HTTP_200_OK,
            message="Image processed successfully",
            result=objects_distances_list
        ).model_dump()

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=ErrorResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            error=str(e)
        ).model_dump())
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
