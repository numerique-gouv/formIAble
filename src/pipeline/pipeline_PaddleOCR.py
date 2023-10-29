# importe des classes du module permettant la compatibilité avec les conseils sur le typage (type hints)
from typing import Dict, Final, List, Optional, Tuple
import os.path
import json
import tempfile
import paddleocr
import cv2
import numpy as np
import src.util.utils as utils
import src.models.auto_rotation_translation.functions as ocrFunctions
import src.models.classify_form.PaddleOCR_TextMatch.classify as ocrExtractor
import logging
import pickle


# --- Constantes ---#
# -- Les constantes suivantes définissent des noms de champs dans le fichier de configuration JSON d'un formulaire CERFA. --#
# nom du champ indiquant les couples "nom du champ à extraire : coordonnées de la boîte l'entourant"
FORM_FIELDS_TO_EXTRACT_FIELD_NAME_STR: Final[str] = "fields_to_extract"
# nom du champ listant les coordonnées (x, y) des boîtes entourant les 3 éléments de texte de référence
# Dans le fichier de configuration, ce nom est donc suivi d'une liste de 3 listes (car 3 éléments de texte de référence) de
# 4 listes (car 4 points définissent chaque boîte) de 2 nombres flottants (les coordonnées x et y de chaque point)
FORM_REFERENCE_BOXES_FIELD_NAME_STR: Final[str] = "reference_boxes"
# nom du champ indiquant la taille (hauteur et largeur sous forme de liste de 2 entiers) de l'image de référence du formulaire
FORM_REFERENCE_SIZE_FIELD_NAME_STR: Final[str] = "reference_size"
# nom du champ listant les 3 éléments de texte de référence à retrouver dans le formulaire
FORM_REFERENCE_TEXTS_FIELD_NAME_STR: Final[str] = "reference_texts"


def get_form_image_text_elements_and_boxes(
    form_image_path_str: str,
    ocr_model: paddleocr.PaddleOCR
) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    r"""Après OCRisation de l'image `form_image_path_str`, retourne :
    - la liste des éléments de texte extraits,
    - la liste des coordonnées des points définissant les boîtes entourant les éléments de texte extraits

    Parameters
    ----------
    form_image_path_str : str
        le chemin de l'image à analyser.
    ocr_model : paddleocr.PaddleOCR
        le modèle PaddleOCR à utiliser pour analyser l'image `form_image_path_str`.

    Returns
    -------
    - list of str
        la liste des éléments de texte extraits.
    - list of lists of tuples of int and int
        la liste des coordonnées des points définissant les boîtes entourant les éléments de texte extraits.
    """
    # résultats de l'OCRisation de l'image form_image_path_str permettant de récupérer les différents éléments de texte
    # extraits ainsi que les coordonnées des boîtes entourant lesdits éléments de texte
    input_image_ocr_results = ocr_model.ocr(img=form_image_path_str, det=True, rec=True, cls=True)
    logging.debug("Nombre de résultats extraits par OCR =", len(input_image_ocr_results))
    # liste des coordonnées des points définissant les boîtes entourant les éléments de texte extraits
    input_document_text_boxes: List[List[Tuple[int, int]]] = [input_image_ocr_result[0] for input_image_ocr_result in input_image_ocr_results]
    # boîtes entourant les éléments de texte extraits
    for lTextBoxIdx, lTextBox in enumerate(input_document_text_boxes):
        logging.debug(f"Boîte entourant un élément de texte extrait n° {lTextBoxIdx} = {lTextBox}")
    # éléments de texte extraits avec score
    for lTextElementAndScoreIdx, lTextElementAndScoreStr in enumerate([input_image_ocr_result[1] for input_image_ocr_result in input_image_ocr_results]):
        logging.debug(f"Texte extrait et score n° {lTextElementAndScoreIdx} = {lTextElementAndScoreStr}")
    # liste des éléments de texte extraits, sans le score
    input_document_text_elements: List[str] = [input_image_ocr_result[1][0] for input_image_ocr_result in input_image_ocr_results]
    del input_image_ocr_results
    return input_document_text_elements, input_document_text_boxes


def get_transformationMatrix_and_save_image_after_affineTransformation(
    input_document_path_str: str,
    form_image_path_str: str,
    form_number_str: str,
    input_document_text_elements: List[str],
    input_document_text_boxes: List[List[Tuple[int, int]]],
    configuration_files_dir_path_str: str
) -> Tuple[str, np.ndarray]:
    r"""A partir de l'image `form_image_path` obtenue du document `input_document_path`, effectue les opérations suivantes :
    - trouve les boîtes entourant les éléments de texte correspondant le mieux aux éléments de texte de l'image de référence
    - applique ensuite une rotation à l'image, en sauvegarde le résultat dans un fichier différent pour éviter tout conflit
      de nommage
    - retourne enfin :
      * le chemin de l'image ainsi obtenue,
      * la matrice de transformation de l'image

    Parameters
    ----------
    input_document_path_str : str
        le chemin du document à analyser.
    form_image_path_str : str
        le chemin de l'image obtenue à partir du document à analyser.
    form_number_str : str
        le numéro CERFA du formulaire détecté dans l'image à analyser `form_image_path_str`.
    input_document_text_elements : list of str
        liste des éléments de texte extraits de l'OCRisation de l'image `form_image_path_str`
    input_document_text_boxes : list of lists of tuples of int and int
        liste des coordonnées des points définissant les boîtes entourant lesdits éléments de texte extraits de l'OCRisation
        de l'image `form_image_path_str`
    configuration_files_dir_path : str
        chemin du répertoire des fichiers de configuration JSON des formulaires CERFA

    Returns
    -------
    - str
        le chemin de l'image transformée.
    - numpy.ndarray
        la matrice de transformation de l'image.
    """
    form_config_file_path: str = os.path.join(configuration_files_dir_path_str, f"cerfa_{form_number_str}.json")
    assert os.path.isfile(
        form_config_file_path
    ), f"Le numéro CERFA {form_number_str} a été extrait du document {input_document_path_str}, " \
       f"mais **aucun fichier de configuration correspondant** {form_config_file_path} n'existe dans le répertoire {configuration_files_dir_path_str}."
    form_config_file_dict: Dict = read_form_config_file(form_config_file_path=form_config_file_path)

    # trouve les boîtes entourant les éléments de texte correspondant le mieux aux éléments de texte de
    # l'image de référence du formulaire
    input_document_matching_boxes = ocrFunctions.find_matching_boxes(
        input_image_text_boxes=input_document_text_boxes,
        input_image_text_elements=input_document_text_elements,
        reference_image_text_elements=form_config_file_dict[FORM_REFERENCE_TEXTS_FIELD_NAME_STR]
    )

    form_image_withoutPrefix_path_str = \
        form_image_path_str.removeprefix(f"{utils.TEMPORARY_FILE_DIRECTORY_STR}/{utils.TEMPORARY_FILE_PREFIX_STR}_")
    transformed_image_path_str: str = f"{utils.TEMPORARY_FILE_DIRECTORY_STR}/auto_transformed_{form_image_withoutPrefix_path_str}"
    # applique une rotation à l'image, en sauvegarde le résultat dans un fichier différent pour éviter
    # tout conflit de nommage et retourne la matrice de transformation de l'image
    transformation_matrix: np.ndarray = ocrFunctions.get_transformationMatrix_and_save_image_after_affineTransformation_with_boxes(
        input_image_path=form_image_path_str,
        output_image_path=transformed_image_path_str,
        input_image_boxes=input_document_matching_boxes,
        reference_image_boxes=form_config_file_dict[FORM_REFERENCE_BOXES_FIELD_NAME_STR],
        output_image_size=form_config_file_dict[FORM_REFERENCE_SIZE_FIELD_NAME_STR]
    )
    return transformed_image_path_str, transformation_matrix


def extract_document(
    input_document_path: str,
    configuration_files_dir_path: str,
    ocr_model: paddleocr.PaddleOCR,
    save_annotated_document: Optional[str] = None
) -> Dict[str, str]:
    """Extract key-value info from a document
    The cerfa number is ocr-ised, then the corresponding configuration file cerfa_*****_**.json
    is searched for in path_dir_configs.
    If save_annotated_document is set, input document is saved after auto-transformation,
    along with reference boxes and extracted boxes [not implemented yet].
    """
    # exporte le document en entrée vers une image au format PNG afin de faciliter les opérations suivantes
    png_document_image, png_document_path_str = utils.get_and_save_image_and_path_from_document(
        input_document_path=input_document_path,
        output_image_format="PNG",
        output_image_quality_in_dpi=350
    )

    # liste des éléments de texte extraits et des coordonnées des points définissant les boîtes entourant lesdits éléments
    # de texte extraits de l'OCRisation de l'image png_document_path_str
    input_document_text_elements, input_document_text_boxes = \
        get_form_image_text_elements_and_boxes(png_document_path_str, ocr_model)

    # extrait le numéro CERFA du formulaire afin de trouver le fichier de configuration dudit formulaire, définissant
    # les champs, leurs positions, les éléments de texte de référence et la taille de l'image de référence associée
    form_number_str: str = ocrExtractor.get_form_number_in_text_elements(
        input_document_path=input_document_path,
        text_elements=input_document_text_elements
    )

    # - trouve les boîtes entourant les éléments de texte correspondant le mieux aux éléments de texte de
    #   l'image de référence du formulaire
    # - applique une rotation à l'image, en sauvegarde le résultat dans un fichier différent pour éviter
    #   tout conflit de nommage et retourne l'image obtenue et la matrice de transformation de ladite image
    transformed_image_path_str, transformation_matrix = get_transformationMatrix_and_save_image_after_affineTransformation(
        input_document_path_str=input_document_path,
        form_image_path_str=png_document_path_str,
        form_number_str=form_number_str,
        input_document_text_elements=input_document_text_elements,
        input_document_text_boxes=input_document_text_boxes,
        configuration_files_dir_path_str=configuration_files_dir_path
    )

    pickle.dump(transformation_matrix, open(f"{utils.TEMPORARY_FILE_DIRECTORY_STR}/transformation_matrix.dump", "wb"))

    # apply transformation to each input box to compare with reference boxes
    transformed_input_boxes: List[List[np.ndarray]] = []
    for input_document_text_box in input_document_text_boxes:
        transformed_input_boxes.append([])
        for corner in input_document_text_box:
            transformed_corner: np.ndarray = transformation_matrix.dot(np.array([corner[0], corner[1], 1]))
            transformed_input_boxes[-1].append(transformed_corner)

    for transformed_box, text_element_str in zip(transformed_input_boxes, input_document_text_elements):
        logging.debug(f"{text_element_str} ----- {transformed_box}")

    os.remove(png_document_path_str)
    os.remove(transformed_image_path_str)
#    raise NotImplementedError


def register_document_as_reference(
    document_to_register_path: str,
    reference_documents_dir_path: str,
    document_to_register_reference_texts: List[str],
    ocr_model: paddleocr.PaddleOCR
) -> None:
    """Registers a document.
    The fields to extract are supposed to be in a configuration file named
    cerfa_*****_**.json in the directory reference_documents_dir_path, where
    the document is supposed to be named cerfa_*****_**
    (see src.util.utils.get_and_save_image_from_document for accepted extensions).
    The document is exported to png and saved in reference_documents_dir_path
    Text elements of document_to_register_reference_texts are detected in the document and added
    in the configuration file cerfa_*****_**.json.
    """
    document_to_register_withoutExtension_basename_str: str = \
        os.path.splitext(os.path.basename(document_to_register_path))[0]
    assert f"{document_to_register_withoutExtension_basename_str}.json" in os.listdir(reference_documents_dir_path), \
        f"Config file {document_to_register_withoutExtension_basename_str}.json not found in {reference_documents_dir_path}"

    reference_config_file_path_str: str = \
        os.path.join(reference_documents_dir_path, f"{document_to_register_withoutExtension_basename_str}.json")
    reference_image_path_str: str = \
        os.path.join(reference_documents_dir_path, f"{document_to_register_withoutExtension_basename_str}.png")
    utils.get_and_save_image_from_document(
        input_document_path=document_to_register_path,
        output_image_path=reference_image_path_str,
        output_image_format="PNG",
        output_image_quality_in_dpi=350
    )

    # Get boxes and size from reference image, to later perform match with input image
    reference_image_ocr_results = ocr_model.ocr(img=reference_image_path_str, det=True, rec=True, cls=False)
    reference_image_allBoxes: List[List[Tuple[int, int]]] = \
        [reference_image_ocr_result[0] for reference_image_ocr_result in reference_image_ocr_results]
    reference_image_allTexts: List[str] = \
        [reference_image_ocr_result[1][0] for reference_image_ocr_result in reference_image_ocr_results]
    del reference_image_ocr_results
    logging.debug(f"All texts found in image {reference_image_path_str}: {reference_image_allTexts}")
    reference_image_boxes: List[List[Tuple[int, int]]] = ocrFunctions.find_matching_boxes(
        input_image_text_boxes=reference_image_allBoxes,
        input_image_text_elements=reference_image_allTexts,
        reference_image_text_elements=document_to_register_reference_texts
    )

    reference_image_size = cv2.imread(reference_image_path_str).shape[:2]  # height, width of image

    with open(reference_config_file_path_str, "r") as reference_config_file:
        reference_config_file_dict = json.load(reference_config_file)

    reference_config_file_dict[FORM_REFERENCE_TEXTS_FIELD_NAME_STR] = document_to_register_reference_texts
    reference_config_file_dict[FORM_REFERENCE_BOXES_FIELD_NAME_STR] = reference_image_boxes
    reference_config_file_dict[FORM_REFERENCE_SIZE_FIELD_NAME_STR] = reference_image_size

    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_reference_config_file:
            json.dump(reference_config_file_dict, temp_reference_config_file)
        os.remove(reference_config_file_path_str)
        os.rename(temp_reference_config_file.name, reference_config_file_path_str)
    except Exception:
        os.remove(temp_reference_config_file.name)
        raise


def reshape_corners_positions(x0, y0, width, height) -> List[Tuple[int, int]]:
    x0, y0, width, height = int(x0), int(y0), int(width), int(height)
    x1, y1 = x0 + width, y0 + height
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def read_form_config_file(form_config_file_path: str) -> Dict:
    """Reads form configuration file.

    Assumed structure of input config file is:
    fields_to_extract:
        field_name_1: positions, i.e. list of 4 floats or list of lists of 4 floats, corresponding to x0, y0, width, height.
        field_name_2: same
        ...
    reference_texts: list of 3 strings
    reference_boxes: list of boxes, i.e. list of 3 lists of 4 lists of 2 float, describing the x, y coordinates of each corner.
    reference_size: list of 2 floats, i.e. height and width of the reference image
    """
    with open(form_config_file_path, "r") as form_json_config_file:
        form_config_file_dict = json.load(form_json_config_file)

    resulting_form_config_file_dict: Dict = {
        FORM_REFERENCE_TEXTS_FIELD_NAME_STR: form_config_file_dict[FORM_REFERENCE_TEXTS_FIELD_NAME_STR],
        FORM_FIELDS_TO_EXTRACT_FIELD_NAME_STR: {},
        FORM_REFERENCE_BOXES_FIELD_NAME_STR: []
    }
    for field_name, field_positions in form_config_file_dict[FORM_FIELDS_TO_EXTRACT_FIELD_NAME_STR].items():
        if isinstance(field_positions[0], float) or isinstance(field_positions[0], int):
            resulting_form_config_file_dict[FORM_FIELDS_TO_EXTRACT_FIELD_NAME_STR][field_name] = \
                reshape_corners_positions(*field_positions)
        elif isinstance(field_positions[0], list):
            resulting_form_config_file_dict[FORM_FIELDS_TO_EXTRACT_FIELD_NAME_STR][field_name] = [
                reshape_corners_positions(*box_positions) for box_positions in field_positions
            ]
    resulting_form_config_file_dict[FORM_REFERENCE_BOXES_FIELD_NAME_STR] = [
        [(int(corner[0]), int(corner[1])) for corner in corners]
            for corners in form_config_file_dict[FORM_REFERENCE_BOXES_FIELD_NAME_STR]
    ]
    reference_height, reference_width = form_config_file_dict[FORM_REFERENCE_SIZE_FIELD_NAME_STR]
    resulting_form_config_file_dict[FORM_REFERENCE_SIZE_FIELD_NAME_STR] = \
        (int(reference_height), int(reference_width))
    return resulting_form_config_file_dict
