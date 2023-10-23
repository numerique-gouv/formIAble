from typing import List, Tuple, Dict, Optional
import os.path
import json
import tempfile
import paddleocr
import cv2
import numpy as np
import src.util.utils as utils
import src.models.auto_rotation_translation.functions
import src.models.classify_form.PaddleOCR_TextMatch.classify
import logging
import pickle


def extract_document(path_document_input: str,
                     path_dir_configs: str,
                     ocr_model: paddleocr.PaddleOCR,
                     save_annotated_document: Optional[str] = None) -> Dict[str, str]:
    """Extract key-value info from a document
    The cerfa number is ocr-ised, then the corresponding configuration file cerfa_*****_**.json
    is searched for in path_dir_configs.
    If save_annotated_document is set,
    input document is saved after auto-transformation, along with reference boxes and extracted boxes.
    """

    # export to png to facilitate subsequent operations
    name_document_input_with_extention_png: str = f"{os.path.splitext(os.path.basename(path_document_input))[0]}.png"
    logging.debug(f"Nom de l'image PNG = {name_document_input_with_extention_png}")
    path_document_png: str = f"temp_{name_document_input_with_extention_png}"
    utils.convert_to_png(path_document_input=path_document_input,
                         path_document_output=path_document_png)

    # ocr input document to get texts and box corners
    ocr_result_image_input = ocr_model.ocr(img=path_document_png, det=True, rec=True, cls=True)
    logging.debug("Nombre de résultats extraits par OCR =", len(ocr_result_image_input))
    # boîtes de contour des textes extraits
    for lBoxIdx, lBox in enumerate([box_and_text[0] for box_and_text in ocr_result_image_input]):
        logging.debug(f"Boîte n° {lBoxIdx} = {lBox}")
    # textes extraits avec score
    for lTextAndScoreIdx, lTextAndScoreStr in enumerate([box_and_text[1] for box_and_text in ocr_result_image_input]):
        logging.debug(f"Texte et score n° {lTextAndScoreIdx} = {lTextAndScoreStr}")
    # liste des boîtes de contour des textes extraits
    boxes_document_input: List[List[Tuple[int, int]]] = [box_and_text[0] for box_and_text in ocr_result_image_input]
    # liste des textes extraits
    texts_document_input: List[str] = [box_and_text[1][0] for box_and_text in ocr_result_image_input]

    # automatically extract reference to find reference config
    reference: str = src.models.classify_form.PaddleOCR_TextMatch.classify.get_reference_from_texts(
        image_path=path_document_input,
        texts=texts_document_input
    )
    logging.debug(f"path_document_input = {path_document_input}, extracted reference {reference}")
    path_config_reference: str = os.path.join(path_dir_configs, f"cerfa_{reference}.json")
    assert os.path.isfile(
        path_config_reference
    ), f"Extracted reference {reference} from document {path_document_input}, " \
       f"did not found config file {path_config_reference} in directory {path_dir_configs}"
    config_reference: Dict = read_config_reference(path_config_reference=path_config_reference)

    # find matching boxes to perform auto-transformation
    matching_boxes_document_input = src.models.auto_rotation_translation.functions.find_matching_boxes(
        boxes_image_input=boxes_document_input,
        texts_image_input=texts_document_input,
        texts_image_reference=config_reference["texts_reference"]
    )

    # Auto rotate and save to another file to avoid file conflicts
    path_image_auto_transformed: str = f"auto_transformed_{name_document_input_with_extention_png}"
    matrix_transformation: np.ndarray = src.models.auto_rotation_translation.functions.affine_transform_from_boxes(
        path_image_input=path_document_png,
        path_image_output=path_image_auto_transformed,
        boxes_image_input=matching_boxes_document_input,
        boxes_image_reference=config_reference["boxes_reference"],
        size_image_output=config_reference["size_reference"]
    )

    pickle.dump(matrix_transformation, open("dump_matrix_transformation", "wb"))

    # Apply transformation to each input box to compare with reference boxes
    boxes_input_after_transform: List[List[np.ndarray]] = []
    for box_document_input in boxes_document_input:
        boxes_input_after_transform.append([])
        for corner in box_document_input:
            corner_transformed: np.ndarray = matrix_transformation.dot(np.array([corner[0], corner[1], 1]))
            boxes_input_after_transform[-1].append(corner_transformed)

    for box_after_transform, text in zip(boxes_input_after_transform, texts_document_input):
        logging.debug(f"{text} ----- {box_after_transform}")

    os.remove(path_document_png)
    os.remove(path_image_auto_transformed)
#    raise NotImplementedError


def register_document(path_document_to_register: str,
                      path_dir_reference_documents: str,
                      texts_reference: List[str],
                      ocr_model: paddleocr.PaddleOCR) -> None:
    """Register a document
    The fields to extract are supposed to be in a configuration file
    named cerfa_*****_**.json in directory path_dir_reference_documents, where
    the document is supposed to be named cerfa_*****_** (see convert_to_png for accepted extensions).
    The document is exported to png and saved in path_dir_reference_documents
    Text elements of texts_reference are detected in the document and added
    in the configuration file cerfa_*****_**.json.
    """
    name_document_to_register: str = os.path.splitext(os.path.basename(path_document_to_register))[0]
    assert f"{name_document_to_register}.json" in os.listdir(path_dir_reference_documents), \
        f"Config file {name_document_to_register}.json not found in {path_dir_reference_documents}"

    path_config: str = os.path.join(path_dir_reference_documents, f"{name_document_to_register}.json")
    path_image_reference: str = os.path.join(path_dir_reference_documents, f"{name_document_to_register}.png")
    utils.convert_to_png(path_document_input=path_document_to_register,
                         path_document_output=path_image_reference)

    # Get boxes and size from reference image, to later perform match with input image
    ocr_result_image_reference = ocr_model.ocr(img=path_image_reference, det=True, rec=True, cls=False)
    all_boxes_image_reference: List[List[Tuple[int, int]]] = [box_and_text[0] for box_and_text
                                                              in ocr_result_image_reference[0]]
    all_texts_image_reference: List[str] = [box_and_text[1][0] for box_and_text in ocr_result_image_reference[0]]
    del ocr_result_image_reference
    logging.debug(f"All texts found in image {path_image_reference}: {all_texts_image_reference}")
    boxes_reference: List[List[Tuple[int, int]]] = src.models.auto_rotation_translation.functions.find_matching_boxes(
        boxes_image_input=all_boxes_image_reference,
        texts_image_input=all_texts_image_reference,
        texts_image_reference=texts_reference
    )

    size_image_reference = cv2.imread(path_image_reference).shape[:2]  # height, width of image

    with open(path_config, "r") as config_file:
        config_content = json.load(config_file)

    config_content["texts_reference"] = texts_reference
    config_content["boxes_reference"] = boxes_reference
    config_content["size_reference"] = size_image_reference

    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as config_file_temp:
            json.dump(config_content, config_file_temp)
        os.remove(path_config)
        os.rename(config_file_temp.name, path_config)
    except Exception:
        os.remove(config_file_temp.name)
        raise


def reshape_corners_positions(x0, y0, width, height) -> List[Tuple[int, int]]:
    x0, y0, width, height = int(x0), int(y0), int(width), int(height)
    x1, y1 = x0 + width, y0 + height
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def read_config_reference(path_config_reference: str) -> Dict:
    """Read config file

    Assumed structure of input config file is:
    fields_to_extract:
        field_name_1: positions (list of 4 float or list of lists of 4 float, corresponding to x0, y0, width, height)
        field_name_2: same
        ...
    texts_reference: list of 3 strings
    boxes_reference: list of boxes (list of 3 lists of 4 lists of 2 float, describing the x, y coordinates of each corner)
    size_reference: list of 2 float (height, width of reference image)
    """
    with open(path_config_reference, "r") as config_file_reference:
        config_reference = json.load(config_file_reference)

    result: Dict = {"texts_reference": config_reference["texts_reference"],
                    "fields_to_extract": {},
                    "boxes_reference": []}
    for field_name, field_positions in config_reference["fields_to_extract"].items():
        if isinstance(field_positions[0], float) or isinstance(field_positions[0], int):
            result["fields_to_extract"][field_name] = reshape_corners_positions(*field_positions)
        elif isinstance(field_positions[0], list):
            result["fields_to_extract"][field_name] = [
                reshape_corners_positions(*box_positions) for box_positions in field_positions
            ]
    result["boxes_reference"] = [[(int(corner[0]), int(corner[1])) for corner in corners]
                                 for corners in config_reference["boxes_reference"]]
    height_reference, width_reference = config_reference["size_reference"]
    result["size_reference"] = (int(height_reference), int(width_reference))
    return result
