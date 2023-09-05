import os.path

import paddleocr
import cv2
import numpy as np
from typing import List
import unicodedata
import Levenshtein
import logging
logging.basicConfig(level=logging.DEBUG)


def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


def auto_rotate_and_translate(path_input_image: str,
                              ocr_model: paddleocr.PaddleOCR,
                              path_output_dir: str,
                              path_reference_image: str,
                              texts_reference: List[str],
                              cls: bool):

    input_image = cv2.imread(path_input_image)
    logging.debug(f"Loaded input image, shape {input_image.shape}")
    ocr_result_input_image = ocr_model.ocr(
        img=path_input_image,
        det=True,
        rec=True,
        cls=cls
    )
    texts_input_image: List[str] = [
        strip_accents(box_and_text[1][0]).strip() for box_and_text in ocr_result_input_image[0]
    ]
    coordinates_input_image: List[List[List[float]]] = [
        box_and_text[0] for box_and_text in ocr_result_input_image[0]
    ]

    reference_image = cv2.imread(path_reference_image)
    logging.debug(f"Loaded reference image, shape {reference_image.shape}")

    ocr_result_reference_image = ocr_model.ocr(
        img=path_reference_image,
        det=True,
        rec=True,
        cls=cls
    )
    texts_reference_image_: List[str] = [
        strip_accents(box_and_text[1][0]).strip() for box_and_text in ocr_result_reference_image[0]
    ]
    coordinates_reference_image_: List[List[List[float]]] = [
        box_and_text[0] for box_and_text in ocr_result_reference_image[0]
    ]

    # find matches between reference text contents and the extracted text from the reference image
    texts_reference_image: List[str] = []
    coordinates_reference_image: List[List[List[float]]] = []

    texts_reference = [strip_accents(text).strip() for text in texts_reference]

    for text_reference in texts_reference:
        distance_closest = np.Inf
        index_closest = 0
        for index_reference_image, text_reference_image in enumerate(texts_reference_image_):
            distance = Levenshtein.distance(text_reference,
                                            text_reference_image)
            if distance < distance_closest:
                distance_closest = distance
                index_closest = index_reference_image
        texts_reference_image.append(texts_reference_image_[index_closest])
        coordinates_reference_image.append(coordinates_reference_image_[index_closest])

    matching_coordinates_input_image: List[List[List[float]]] = []

    for text_reference_image in texts_reference_image:
        logging.debug(f"Searching in input image text {text_reference_image}")
        distance_closest = np.Inf
        index_closest = 0
        for index_input_image, text_input_image in enumerate(texts_input_image):
            distance = Levenshtein.distance(text_reference_image,
                                            text_input_image)
            if distance < distance_closest:
                distance_closest = distance
                index_closest = index_input_image
        logging.debug(f"Searching in input image text {text_reference_image} : best match found {texts_input_image[index_closest]}")
        matching_coordinates_input_image.append(coordinates_input_image[index_closest])

    assert len(matching_coordinates_input_image) == len(texts_reference)

    input_pts = np.array(matching_coordinates_input_image, dtype=np.float32).mean(axis=1)
    output_pts = np.array(coordinates_reference_image, dtype=np.float32).mean(axis=1)

    assert input_pts.shape == (len(texts_reference), 2)
    assert output_pts.shape == (len(texts_reference), 2)

    matrix_affine_transform = cv2.getAffineTransform(input_pts, output_pts)
    output_image = cv2.warpAffine(input_image,
                                  matrix_affine_transform,
                                  dsize=(reference_image.shape[1], reference_image.shape[0]))
    path_output_image: str = os.path.join(path_output_dir, os.path.basename(path_input_image))

    cv2.imwrite(filename=path_output_image, img=output_image)


ocrModel: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                    lang='fr')

auto_rotate_and_translate(path_input_image="data/synthetic_forms/cerfa_12485_03_fake1.jpg",
                          ocr_model=ocrModel,
                          path_output_dir="results/PaddleOCR/auto-rotation-translation",
                          path_reference_image="data/empty_forms/non-editable/cerfa_12485_03.png",
                          texts_reference=["cerfa",
                                           "S 3704b",
                                           "Déclaration signée le"],
                          cls=True)
