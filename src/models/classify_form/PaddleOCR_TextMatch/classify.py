# importe des classes du module permettant la compatibilité avec les conseils sur le typage (type hints)
from typing import Final, List
# importe le module des paramètres et fonctions systèmes
import sys
import logging
# importe le module des expressions régulières
import re
# importe le module de boîtes à outils d'OCRisation basées sur PaddlePaddle (PArallel Distributed Deep LEarning),
# la seule plateforme chinoise indépendante de deep learning pour la R&D, ouverte à la communauté open source
# depuis 2016
import paddleocr


# expression régulière du numéro CERFA d'un formulaire
CERFA_REFERENCE_REGEXP: Final[str] = r"\d{5}\s?\*\s?\d{2}\s*$"


def get_form_reference(input_document_path: str, ocrModel: paddleocr.PaddleOCR) -> str:
    """
    Extracts CERFA matching references from image obtained from form document input_document_path
    If multiple no reference or multiple references are found, raise ValueError

    :param input_document_path: path to the form document
    :type input_document_path: str
    :param ocrModel: model to use for text detection and recognition
    :type ocrModel: paddleocr.PaddleOCR
    :return: reference text
    :rtype: str
    """
    image_ocr_results = ocrModel.ocr(
        img=input_document_path,
        det=True,
        rec=True,
        cls=True
    )
    text_elements = [image_ocr_result[1][0] for image_ocr_result in image_ocr_results]
    return get_form_number_in_text_elements(input_document_path=input_document_path, text_elements=text_elements)


def get_form_number_in_text_elements(input_document_path: str, text_elements: List[str]) -> str:
    matching_text_elements = []
    logging.debug(f"Eléments de texte dans lesquels chercher des correspondances à la regexp {CERFA_REFERENCE_REGEXP} = " + " ".join(text_elements))
    for text_element in text_elements:
        logging.debug(f"Recherche de correspondances de l'expression régulière {CERFA_REFERENCE_REGEXP} dans le texte {text_element}")
        match = re.search(CERFA_REFERENCE_REGEXP, text_element)
        if match:
            matching_text_elements.append(match.group(0).replace(" ", "").replace("*", "_").strip())
            logging.debug("Correspondances courantes = " + " ".join(matching_text_elements))
    if len(matching_text_elements) == 0:
        raise ValueError(f"Aucun numéro de formulaire CERFA n'a été trouvé dans le document {input_document_path}")
    if len(matching_text_elements) > 1:
        raise ValueError(f"Plusieurs numéros de formulaires CERFA ont été trouvés dans le document {input_document_path} : {matching_text_elements}")
    form_number_str = matching_text_elements[0]
    logging.debug(f"=> Numéro CERFA extrait du document {input_document_path} = {form_number_str}")
    return form_number_str


def add_form_reference(
    form_image_path: str,
    accepted_references_path: str,
    ocrModel: paddleocr.PaddleOCR
):
    """
    Extract the reference from the form image defined in form_image_path and add it to the file of the accepted references.

    :param form_image_path: the path to the image to extract the reference from
    :type form_image_path: str
    :param accepted_references_path: the path to the file containing the accepted references
    :type accepted_references_path: str
    :return: the extracted form reference
    :rtype: str
    """
    form_reference = get_form_reference(form_image_path, ocrModel)
    with open(accepted_references_path, "r+") as accepted_references_file:
        accepted_references_file_content: str = accepted_references_file.read()
        if form_reference in accepted_references_file_content:
            logging.warning(f"Extracted reference {form_reference} already in {accepted_references_path}: no change has been made.")
            return
        if accepted_references_file_content.endswith("\n"):
            accepted_references_file.write(f"{form_reference}\n")
        else:
            accepted_references_file.write(f"\n{form_reference}\n")
    return form_reference


def classify_form_image(
    form_image_path: str,
    accepted_references_path: str,
    ocrModel: paddleocr.PaddleOCR
) -> str:
    """
    Classify the form image defined in form_image_path as an accepted reference or as unmatched

    :param form_image_path: path to image
    :type form_image_path: str
    :param accepted_references_path: path to the file containing the accepted references
    :type accepted_references_path: str
    :return: an accepted reference or "UNMATCHED"
    :rtype: str
    """
    form_reference = get_form_reference(form_image_path, ocrModel)
    with open(accepted_references_path, "r") as accepted_references_file:
        if form_reference in accepted_references_file.read():
            return form_reference
        else:
            return "UNMATCHED"


if __name__ == "__main__":
    action = sys.argv[1]
    ocrModel: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                        lang='fr')
    if action == "get_form_reference":
        if len(sys.argv) != 3:
            raise ValueError("Argument form_image_path is expected for action get_form_reference, no other arguments should be provided.")
        image_path = sys.argv[2]
        print(get_form_reference(image_path, ocrModel))
    elif action == "add_form_reference":
        if len(sys.argv) != 4:
            raise ValueError("Arguments form_image_path and accepted_references_path are expected for action add_form_reference, no other arguments should be provided.")
        image_path = sys.argv[2]
        accepted_references_path = sys.argv[3]
        added_reference = add_form_reference(image_path, accepted_references_path, ocrModel)
        if added_reference:
            print(f"Added reference {added_reference} to {accepted_references_path}")
    elif action == "classify_form_image":
        if len(sys.argv) != 4:
            raise ValueError("Arguments form_image_path and accepted_references_path are expected for action classify_form_image, no other arguments should be provided.")
        image_path = sys.argv[2]
        accepted_references_path = sys.argv[3]
        print(classify_form_image(image_path, accepted_references_path, ocrModel))
    else:
        raise ValueError(f"Unknown action {action}, "
                         "accepted actions are get_form_reference, add_form_reference and classify_form_image")
