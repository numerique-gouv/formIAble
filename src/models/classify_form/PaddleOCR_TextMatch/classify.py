import sys
import logging
import re
import paddleocr

CERFA_REFERENCE_REGEX: str = r"\d{5}\s?\*\s?\d{2}\s*$"  # passed to re.search


def get_reference(image_path: str,
                  ocrModel: paddleocr.PaddleOCR) -> str:
    """
    Extract CERFA matching references from image defined in image_path
    If multiple no reference or multiple references are found, raise ValueError

    :param image_path: path to image
    :type image_path: str
    :param ocrModel: model to use for text detection and recognition
    :type ocrModel: paddleocr.PaddleOCR
    :return: reference text
    :rtype: str
    """
    ocr_result = ocrModel.ocr(
        img=image_path,
        det=True,
        rec=True,
        cls=True
    )
    text_contents = [text_and_score[1][0] for text_and_score in ocr_result[0]]
    text_matching_reference = []
    for text in text_contents:
        match = re.search(CERFA_REFERENCE_REGEX, text)
        if match:
            text_matching_reference.append(match.group(0).replace(" ", "").replace("*", "_").strip())
    if len(text_matching_reference) == 0:
        raise ValueError(f"No CERFA reference found in {image_path}")
    if len(text_matching_reference) > 1:
        raise ValueError(f"Multiple CERFA references found in {image_path}: {text_matching_reference}")
    return text_matching_reference[0]


def add_reference(image_path: str,
                  path_references_accepted: str,
                  ocrModel: paddleocr.PaddleOCR):
    """
    Extract reference from image defined in image_path and add it in accepted references file

    :param image_path: path to image
    :type image_path: str
    :param path_references_accepted: path to file containing accepted references
    :type path_references_accepted: str
    :return: the extracted reference
    :rtype: str
    """
    reference = get_reference(image_path, ocrModel)
    with open(path_references_accepted, "r+") as file_references_accepted:
        file_references_content: str = file_references_accepted.read()
        if reference in file_references_content:
            logging.warning(f"Extracted reference {reference} already in {path_references_accepted}, no change has been made.")
            return
        if file_references_content.endswith("\n"):
            file_references_accepted.write(f"{reference}\n")
        else:
            file_references_accepted.write(f"\n{reference}\n")
    return reference


def classify_image(image_path: str,
                   path_references_accepted: str,
                   ocrModel: paddleocr.PaddleOCR) -> str:
    """
    Classify image defined in image_path into an accepted reference or as unmatched

    :param image_path: path to image
    :type image_path: str
    :param path_references_accepted: path to file containing accepted references
    :type path_references_accepted: str
    :return: An accepted reference or "UNMATCHED"
    :rtype: str
    """
    reference = get_reference(image_path, ocrModel)
    with open(path_references_accepted, "r") as file_references_accepted:
        if reference in file_references_accepted.read():
            return reference
        else:
            return "UNMATCHED"


if __name__ == "__main__":

    action = sys.argv[1]
    ocrModel: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                        lang='fr')
    if action == "get_reference":
        if len(sys.argv) != 3:
            raise ValueError("Argument image_path is expected for action get_reference, no other arguments should be provided.")
        image_path = sys.argv[2]
        print(get_reference(image_path, ocrModel))
    elif action == "add_reference":
        if len(sys.argv) != 4:
            raise ValueError("Arguments image_path and path_references_accepted are expected for action add_reference, no other arguments should be provided.")
        image_path = sys.argv[2]
        path_references_accepted = sys.argv[3]
        added_reference = add_reference(image_path, path_references_accepted, ocrModel)
        if added_reference:
            print(f"Added reference {added_reference} to {path_references_accepted}")
    elif action == "classify_image":
        if len(sys.argv) != 4:
            raise ValueError("Arguments image_path and path_references_accepted are expected for action classify_image, no other arguments should be provided.")
        image_path = sys.argv[2]
        path_references_accepted = sys.argv[3]
        print(classify_image(image_path, path_references_accepted, ocrModel))
    else:
        raise ValueError(f"Unknown action {action}, "
                         "accepted actions are get_reference, add_reference and classify_image")
