import paddleocr
import cv2
import numpy as np
from typing import List
import Levenshtein

cv2.getRotationMatrix2D()

def auto_rotate_and_translate(path_input_image: str,
                              ocrModel: paddleocr.PaddleOCR,
                              path_output_dir: str,
                              path_reference_image: str,
                              reference_text_contents: List[str]):
    ocr_result_input_image = ocrModel.ocr(
        img=path_input_image,
        det=True,
        rec=True,
        cls=True
    )
    text_contents_input_image: List[str] = [box_and_text[1][0] for box_and_text in ocr_result_input_image[0]]
    coordinates_input_image: List[List[List[float]]] = [box_and_text[0] for box_and_text in ocr_result_input_image[0]]

    ocr_result_reference_image = ocrModel.ocr(
        img=path_reference_image,
        det=True,
        rec=True,
        cls=False
    )
    text_contents_reference_image: List[str] = [box_and_text[1][0] for box_and_text in ocr_result_reference_image[0]]
    coordinates_reference_image: List[List[List[float]]] = [box_and_text[0] for box_and_text in ocr_result_reference_image[0]]

    for text_content_reference_image in text_contents_reference_image:
        closest_distance = np.Inf
        closest_text_content_input_image = None
        for text_content_input_image in text_contents_input_image:
            distance = Levenshtein.distance(text_content_reference_image.strip().lower(),
                                            text_content_input_image.strip().lower())
            if distance < closest_distance:
                closest_distance = distance
                closest_text_content_input_image = text_content_input_image



    # todo: once match is found, store both box center coordinates

    # todo: use cv2.getAffineTransform to get transformation matrix
    # todo: then use cv2.warpAffine to apply transformation matrix to input image
    # todo: and then crop to get the same dimensions as reference image





ocrModel: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                    lang='fr')

auto_rotate_and_translate(path_input_image="data/synthetic_forms/cerfa_14011_03_fake2.jpg",
                          ocrModel=ocrModel,
                          path_output_dir="results/PaddleOCR/auto-rotation-translation",
                          path_reference_image="data/empty_forms/non-editable/cerfa_14011_03.png",
                          reference_text_contents=["cerfa",
                                                   "N° 14011*02",
                                                   "Cette déclaration ne vaut pas document d'identité.",
                                                   "Toute fausse déclaration est passible des peines prévues par les articles 441-6 et 441-7 du nouveau code pénal.",
                                                   "Partie réservée á l'administration"])
