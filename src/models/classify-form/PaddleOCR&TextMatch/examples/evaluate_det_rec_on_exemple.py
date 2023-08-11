from __future__ import annotations

import logging
import paddleocr
import os
from PIL import Image, ImageDraw
logging.basicConfig(level=logging.INFO)

example_image_path: str = "data/synthetic_forms/cerfa_12485_03_fake1.jpg"
output_path: str = "results/PaddleOCR/"

# path to save paddleocr image and text output
image_file_name: str = os.path.basename(os.path.splitext(example_image_path)[0])
ocrised_image_path: str = os.path.join(output_path, "".join((image_file_name, ".jpg")))
ocrised_text_path: str = os.path.join(output_path, "".join((image_file_name, ".txt")))

ocrModel = paddleocr.PaddleOCR(use_angle_cls=False, lang='fr')
ocr_result: list[list[list[list[float]] | tuple[str, float]]] = ocrModel.ocr(
    example_image_path,
    cls=False
)

logging.info(f"ocr_result = \n{ocr_result}")

os.system(f"cp {example_image_path} {ocrised_image_path}")
with Image.open(ocrised_image_path) as image:
    draw = ImageDraw.Draw(image)
    text_file = open(ocrised_text_path, "w")
    # draw detected text areas in red
    for box_coordinates, text_and_score in ocr_result[0]:
        points: list[tuple] = [tuple(point) for point in box_coordinates]
        points.append(points[0])
        draw.line(xy=points, fill="red")
        text_file.write(text_and_score[0] + "\n")

    image.save(ocrised_image_path, format="JPEG")
    text_file.close()

