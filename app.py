import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from img2glb import load_zoe_model, get_mesh

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

try:
    model = load_zoe_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.post("/generate_mesh")
async def generate_mesh(
    image: UploadFile = File(...),
    keep_edges: bool = Form(False)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not image.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    filename = image.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_filename = f"{os.path.splitext(filename)[0]}.glb"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)

        logger.info(f"Processing image: {input_path}")
        get_mesh(model, input_path, output_path, keep_edges=keep_edges)
        logger.info(f"Mesh generated successfully: {output_path}")

        # Return the file
        response = FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=output_filename
        )
        delete_input_file(input_path)

        return response

    except Exception as e:
        logger.error(f"Error generating mesh: {str(e)}", exc_info=True)
        try:
            os.remove(input_path)
            logger.info(f"Deleted input file due to error: {input_path}")
        except Exception as del_e:
            logger.error(f"Error deleting input file: {str(del_e)}")
        raise HTTPException(status_code=500, detail=str(e))

def delete_input_file(input_path: str):
    try:
        os.remove(input_path)
        logger.info(f"Deleted input file: {input_path}")
    except Exception as e:
        logger.error(f"Error deleting input file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)