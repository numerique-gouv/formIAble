from typing import List, Tuple
import os.path
import json
import tempfile
import paddleocr
import src.util.utils as utils
import src.models.auto_rotation_translation.functions
import logging


def extract_document(path_input_document: str,
                     path_dir_configs: str):
    raise NotImplementedError


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

    with open(path_config, "r") as config_file:
        config_content = json.load(config_file)

    config_content["reference_texts"] = texts_reference
    config_content["reference_boxes"] = boxes_reference

    try:
        with tempfile.NamedTemporaryFile(mode="w",delete=False) as config_file_temp:
            json.dump(config_content, config_file_temp)
        os.remove(path_config)
        os.rename(config_file_temp.name, path_config)
    except Exception:
        os.remove(config_file_temp.name)
        raise
