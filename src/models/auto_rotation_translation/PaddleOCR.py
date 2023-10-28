from typing import List, Tuple
import paddleocr
import cv2
import logging
# importe le module des classes représentant le système de fichiers avec la sémantique appropriée pour différents
# systèmes d'exploitation (chemins orientés objet)
from pathlib import Path
# importe le module des paramètres et fonctions systèmes
import sys
# se positionne sur le chemin de travail courant
cwd = Path().resolve()
# ajoute le chemin de travail courant à la variable concaténant les répertoires système afin de permettre l'import
# de modules présents dans les sous-répertoires dudit répertoire
sys.path.append(str(cwd))
import src.models.auto_rotation_translation.functions as ocrFunctions

logging.basicConfig(level=logging.DEBUG)


def auto_rotate_and_translate(input_image_path: str,
                              ocr_model: paddleocr.PaddleOCR,
                              output_image_path: str,
                              reference_image_path: str,
                              reference_image_texts: List[str],
                              cls: bool):
    # Get boxes and size from reference image, to later perform match with input image
    reference_image_ocr_results = ocr_model.ocr(img=reference_image_path, det=True, rec=True, cls=False)
    reference_image_allBoxes: List[List[Tuple[int, int]]] = \
        [reference_image_ocr_result[0] for reference_image_ocr_result in reference_image_ocr_results]
    reference_image_allTexts: List[str] = \
        [reference_image_ocr_result[1][0] for reference_image_ocr_result in reference_image_ocr_results]
    del reference_image_ocr_results
    logging.debug(f"All texts found in image {reference_image_path}: {reference_image_allTexts}")
    reference_image_boxes: List[List[Tuple[int, int]]] = ocrFunctions.find_matching_boxes(
        input_image_text_boxes=reference_image_allBoxes,
        input_image_text_elements=reference_image_allTexts,
        reference_image_text_elements=reference_image_texts
    )

    reference_image = cv2.imread(reference_image_path)
    reference_image_shape: Tuple[int, int] = reference_image.shape

    input_image_ocr_results = ocr_model.ocr(img=input_image_path, det=True, rec=True, cls=cls)

    input_image_boxes: List[List[Tuple[int, int]]] = \
        [input_image_ocr_result[0] for input_image_ocr_result in input_image_ocr_results]
    input_image_texts: List[str] = \
        [input_image_ocr_result[1][0] for input_image_ocr_result in input_image_ocr_results]
    del input_image_ocr_results

    input_image_matching_boxes: List[List[Tuple[int, int]]] = ocrFunctions.find_matching_boxes(
        input_image_text_boxes=input_image_boxes,
        input_image_text_elements=input_image_texts,
        reference_image_text_elements=reference_image_texts
    )

    ocrFunctions.get_transformationMatrix_and_save_image_after_affineTransformation_with_boxes(
        input_image_path=input_image_path,
        output_image_path=output_image_path,
        input_image_boxes=input_image_matching_boxes,
        reference_image_boxes=reference_image_boxes,
        output_image_size=reference_image_shape
    )


ocrModel: paddleocr.PaddleOCR = paddleocr.PaddleOCR(
    use_angle_cls=True,
    lang='fr'
)

auto_rotate_and_translate(
    input_image_path="./data/synthetic_forms/cerfa_12485_03_fake1.jpg",
    ocr_model=ocrModel,
    output_image_path="./data/tmp/transformed_cerfa_12485_03_fake1.jpg",
    reference_image_path="./data/empty_forms/non-editable/cerfa_12485_03.png",
    reference_image_texts=[
        "cerfa",
        "S 3704b",
        "Déclaration signée le"
    ],
    cls=True
)
