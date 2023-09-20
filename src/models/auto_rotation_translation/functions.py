from typing import List, Tuple, Iterator
import unicodedata
import Levenshtein
import numpy as np
import cv2
import logging


def affine_transform_from_boxes(path_image_input: str,
                                path_image_output: str,
                                boxes_image_input: List[List[Tuple[int, int]]],
                                boxes_image_reference: List[List[Tuple[int, int]]],
                                size_image_output: Tuple[int, int]) -> np.ndarray:
    """
    Apply the affine transform that matches the centers of boxes_image_input and boxes_image_reference

    :param path_image_input: path of the image to transform
    :param path_image_output: path to save the transformed image
    :param boxes_image_input: list of 3 boxes found in the image to transform used to compute affine transform
    :param boxes_image_reference: list of 3 boxes found in the reference image used to compute affine transform
    :param size_image_output: (height: int, width: int) of the image after transform

    :returns: None
    """
    logging.debug(f"Apply affine transform on image {path_image_input} ...")
    points_image_input: np.ndarray = np.array(boxes_image_input, dtype=np.float32).mean(axis=1)
    assert points_image_input.shape == (3, 2)
    points_image_reference: np.ndarray = np.array(boxes_image_reference, dtype=np.float32).mean(axis=1)
    assert points_image_reference.shape == (3, 2)

    matrix_affine_transform = cv2.getAffineTransform(points_image_input, points_image_reference)
    image_output = cv2.warpAffine(cv2.imread(path_image_input),
                                  matrix_affine_transform,
                                  dsize=(size_image_output[1], size_image_output[0]))
    cv2.imwrite(filename=path_image_output, img=image_output)
    logging.debug(f"Apply affine transform on image {path_image_input} [OK], wrote result to {path_image_output}")
    return matrix_affine_transform


def strip_accents(text):
    """
    Strip accents from input string

    :param text: input string

    :returns: input string without accents.
    """
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


def find_matching_boxes(boxes_image_input: List[List[Tuple[int, int]]],
                        texts_image_input: List[str],
                        texts_image_reference: List[str]) -> List[List[Tuple[int, int]]]:
    """
    Return box coordinates of the nearest matches in texts_image_input for each text in texts_image_references

    :param boxes_image_input: the list of corners of each box found in image_input
    :param texts_image_input: the list of extracted texts of each box in image_input
    :param texts_image_reference: texts in image_reference to look for

    :returns: the list of corners of each box whose text best matches the corresponding text in texts_image_reference
    """
    assert len(boxes_image_input) == len(texts_image_input)
    matching_boxes_image_input: List[List[Tuple[int, int]]] = []
    for text_image_reference in map(lambda s: strip_accents(s).strip(), texts_image_reference):
        distance_best_match = np.Inf
        index_best_match = 0
        # logging.debug(f"Searching for text {text_image_reference}")
        for index_image_input, text_image_input in enumerate(map(lambda s: strip_accents(s).strip(),
                                                                 texts_image_input)):
            assert isinstance(text_image_reference, str)
            assert isinstance(text_image_input, str)
            distance = Levenshtein.distance(text_image_reference,
                                            text_image_input)
            # logging.debug(f"Levenshtein distance between {text_image_reference} and {text_image_input} = {distance}")
            if distance < distance_best_match:
                distance_best_match = distance
                index_best_match = index_image_input
        logging.debug(f"Searching for text {text_image_reference} : best match found {texts_image_input[index_best_match]}")
        matching_boxes_image_input.append(boxes_image_input[index_best_match])

    assert len(matching_boxes_image_input) == len(texts_image_reference)
    return matching_boxes_image_input
