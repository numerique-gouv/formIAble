from typing import List, Tuple, Iterator
import unicodedata
import Levenshtein
import numpy as np
import cv2
import logging


def get_transformationMatrix_and_save_image_after_affineTransformation_with_boxes(
    input_image_path: str,
    output_image_path: str,
    input_image_boxes: List[List[Tuple[int, int]]],
    reference_image_boxes: List[List[Tuple[int, int]]],
    output_image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Gets the transformation matrix and saves the image obtained after having applied the affine transformation that matches
    the centers of input_image_boxes and reference_image_boxes.

    :param input_image_path: path of the image to transform.
    :param output_image_path: path to use to save the transformed image.
    :param input_image_boxes: list of 3 boxes found in the image to transform used to compute the affine transformation.
    :param reference_image_boxes: list of 3 boxes found in the reference image used to compute the affine transformation.
    :param output_image_size: size (height: int, width: int) of the image after transformation.

    :returns: the transformation matrix.
    """
    logging.debug(f"Applying affine transformation on image {input_image_path}...")
    input_image_points: np.ndarray = np.array(input_image_boxes, dtype=np.float32).mean(axis=1)
    # il doit y avoir 3 points ayant 2 coordonnées correspondant aux centres des 3 boîtes
    assert input_image_points.shape == (3, 2)
    reference_image_points: np.ndarray = np.array(reference_image_boxes, dtype=np.float32).mean(axis=1)
    # il doit y avoir 3 points ayant 2 coordonnées correspondant aux centres des 3 boîtes
    assert reference_image_points.shape == (3, 2)

    affine_transformation_matrix = cv2.getAffineTransform(input_image_points, reference_image_points)
    output_image = cv2.warpAffine(cv2.imread(input_image_path),
                                  affine_transformation_matrix,
                                  dsize=(output_image_size[1], output_image_size[0]))
    cv2.imwrite(filename=output_image_path, img=output_image)
    logging.debug(f"Affine transformation applied [OK] on image {input_image_path}, wrote resulting image to {output_image_path}")
    return affine_transformation_matrix


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


def find_matching_boxes(
    input_image_text_boxes: List[List[Tuple[int, int]]],
    input_image_text_elements: List[str],
    reference_image_text_elements: List[str]
) -> List[List[Tuple[int, int]]]:
    """
    Return box coordinates of the nearest matches in input_image_text_elements for each text in reference_image_text_elements

    :param input_image_text_boxes: the list of the corners of each box found in the input image
    :param input_image_text_elements: the list of the extracted texts in each box found in the input image
    :param reference_image_text_elements: text elements to look for in the reference image

    :returns: the list of corners of each box whose text best matches the corresponding text in reference_image_text_elements
    """
    assert len(input_image_text_boxes) == len(input_image_text_elements)
    input_image_matching_boxes: List[List[Tuple[int, int]]] = []
    for reference_image_text_str in map(lambda lString: strip_accents(lString).strip(), reference_image_text_elements):
        best_match_distance = np.Inf
        best_match_index = 0
        # logging.debug(f"Searching for text {reference_image_texts}")
        for input_image_text_index, input_image_text_str in enumerate(
            map(
                lambda lString: strip_accents(lString).strip(),
                input_image_text_elements
            )
        ):
            assert isinstance(reference_image_text_str, str)
            assert isinstance(input_image_text_str, str)
            distance = Levenshtein.distance(reference_image_text_str,
                                            input_image_text_str)
            # logging.debug(f"Levenshtein distance between {reference_image_text} and {input_image_text_str} = {distance}")
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_index = input_image_text_index
        logging.debug(f"Searching for text {reference_image_text_str} : best match found = {input_image_text_elements[best_match_index]}")
        input_image_matching_boxes.append(input_image_text_boxes[best_match_index])

    assert len(input_image_matching_boxes) == len(reference_image_text_elements)
    return input_image_matching_boxes
