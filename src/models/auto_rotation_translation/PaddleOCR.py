from typing import List, Tuple
import paddleocr
import cv2
import logging
import src.models.auto_rotation_translation.functions as functions

logging.basicConfig(level=logging.DEBUG)


def auto_rotate_and_translate(path_image_input: str,
                              ocr_model: paddleocr.PaddleOCR,
                              path_image_output: str,
                              path_image_reference: str,
                              texts_image_reference: List[str],
                              cls: bool):
    # Get boxes and size from reference image, to later perform match with input image
    ocr_result_image_reference = ocr_model.ocr(img=path_image_reference, det=True, rec=True, cls=False)
    all_boxes_image_reference: List[List[Tuple[int, int]]] = [box_and_text[0] for box_and_text
                                                              in ocr_result_image_reference[0]]
    all_texts_image_reference: List[str] = [box_and_text[1][0] for box_and_text in ocr_result_image_reference[0]]
    del ocr_result_image_reference
    logging.debug(f"All texts found in image {path_image_reference}: {all_texts_image_reference}")
    boxes_image_reference: List[List[Tuple[int, int]]] = functions.find_matching_boxes(
        boxes_image_input=all_boxes_image_reference,
        texts_image_input=all_texts_image_reference,
        texts_image_reference=texts_image_reference
    )

    image_reference = cv2.imread(path_image_reference)
    shape_image_reference: Tuple[int, int] = image_reference.shape

    ocr_result_image_input = ocr_model.ocr(img=path_image_input, det=True, rec=True, cls=cls)

    boxes_image_input: List[List[Tuple[int, int]]] = [box_and_text[0] for box_and_text in ocr_result_image_input[0]]
    texts_image_input: List[str] = [box_and_text[1][0] for box_and_text in ocr_result_image_input[0]]
    del ocr_result_image_input

    matching_boxes_image_input: List[List[Tuple[int, int]]] = functions.find_matching_boxes(boxes_image_input,
                                                                                            texts_image_input,
                                                                                            texts_image_reference)

    functions.affine_transform_from_boxes(path_image_input=path_image_input,
                                          path_image_output=path_image_output,
                                          boxes_image_input=matching_boxes_image_input,
                                          boxes_image_reference=boxes_image_reference,
                                          size_image_output=shape_image_reference)


ocrModel: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                    lang='fr')

auto_rotate_and_translate(path_image_input="data/synthetic_forms/cerfa_12485_03_fake1.jpg",
                          ocr_model=ocrModel,
                          path_image_output="results/PaddleOCR/auto_rotation_translation/cerfa_12485_03_fake1.jpg",
                          path_image_reference="data/empty_forms/non-editable/cerfa_12485_03.png",
                          texts_image_reference=["cerfa",
                                                 "S 3704b",
                                                 "Déclaration signée le"],
                          cls=True)
