"""
First pipeline that takes an image as input and returns an
extraction of the Cerfa form if the template exists in our
database.
"""
import sys
from typing import Dict, Tuple
import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from models.classify_form.doctr.identify_cerfa_doctr import get_list_words_in_page
from util.dataGeneration.writer_13753_04 import Writer13753_04
from util.dataGeneration.writer_13969_01 import Writer13969_01
import fitz
from PIL import Image


AREA_RATIO_THRESHOLD = 0.5
available_cerfas_editable = ["13753*04", "13969*01"]
available_cerfas_non_editable = ["12485_03", "14011_03"]
cerfa_number_to_writer_id = {
    "13753*04": "13753_04",
    "13969*01": "13969_01"
}
writers = {
    "13753_04": Writer13753_04,
    "13969_01": Writer13969_01
}


def get_cerfa_template(cerfa_number: str):
    if cerfa_number in available_cerfas_editable:
        writer_id = cerfa_number_to_writer_id[cerfa_number]
        writer = writers[writer_id](num_cerfa=writer_id)
        writer.fill_form()
        return writer.annotator.dic
    elif cerfa_number in available_cerfas_non_editable:
        raise NotImplementedError("Non editable cerfas, to implement.")
    else:
        raise ValueError(f"Cerfa {cerfa_number} does not have a template yet")


def load_cerfa(cerfa_path: str):
    doc = fitz.open(cerfa_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()

    mode = "RGB"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return image


def ocrize(cerfa_path: str, ocr_engine="Doctr"):
    doc_doctr = DocumentFile.from_pdf(cerfa_path)

    if ocr_engine == "Doctr":
        doctr_model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        output = doctr_model(doc_doctr)
    else:
        raise ValueError(f"OCR engine {ocr_engine} not supported.")

    return output


def identify_cerfa(ocr_output):
    page = ocr_output.pages[0]
    page_content = ""
    for block in page.blocks:
        for line in block.lines:
            content = " ".join([word.value for word in line.words])
            page_content += content + "\n"

    pattern = r"N[°o] +(\d{5}\*\d{2})"
    match = re.search(pattern, page_content, re.IGNORECASE)

    if match:
        result = match.group(1)
        return result
    else:
        raise ValueError("No cerfa number found.")


def process_doctr_output(doctr_output, width: int, height: int):
    processed_output = {}

    page = doctr_output.pages[0]  # docs à une page
    list_words_in_page = get_list_words_in_page(page)

    for word in list_words_in_page:
        label = word.value
        x0, y0 = word.geometry[0][0] * width, word.geometry[0][1] * height
        x1, y1 = word.geometry[1][0] * width,  word.geometry[1][1] * height

        processed_output[(x0, y0, x1, y1)] = label

    return processed_output


def compute_box_area(box: Tuple):
    x0, y0, x1, y1 = box
    box_width = x1 - x0
    box_height = y1 - y0
    box_area = box_width * box_height
    return box_area


def compute_boxes_intersection(box1: Tuple, box2: Tuple):
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    # Calculate the coordinates of the intersection box
    x0_intersection = max(x0_1, x0_2)
    y0_intersection = max(y0_1, y0_2)
    x1_intersection = min(x1_1, x1_2)
    y1_intersection = min(y1_1, y1_2)

    # Check if the intersection box is valid
    if x0_intersection <= x1_intersection and y0_intersection <= y1_intersection:
        return (x0_intersection, y0_intersection, x1_intersection, y1_intersection)
    else:
        return None  # No intersection


def clean_cerfa_template(cerfa_template: Dict):
    clean_template = {}
    # For now cerfa template is the raw writer's annotator dict
    for box, field in cerfa_template.items():
        pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
        numbers = [float(match) for match in re.findall(pattern, box)]
        if len(numbers) == 4:
            x0, y0, x1, y1 = numbers
        else:
            raise ValueError("Could not get coordinates of bounding box.")

        field_name, _ = field
        clean_template[field_name] = (x0, y0, x1, y1)
    return clean_template


def match_bounding_boxes_to_template(
    ocr_bounding_boxes: Dict,
    clean_cerfa_template: Dict,
    area_ratio_threshold: float
):
    filled_template = {key: [] for key in clean_cerfa_template.keys()}
    matched_boxes = {key: [] for key in clean_cerfa_template.keys()}
    for ocr_box, word in ocr_bounding_boxes.items():
        box_area = compute_box_area(ocr_box)
        field_to_increment = None
        max_intersection_area = 0
        for field_name, template_box in clean_cerfa_template.items():
            intersection = compute_boxes_intersection(ocr_box, template_box)
            if intersection is not None:
                intersection_area = compute_box_area(intersection)
                if (intersection_area / box_area > area_ratio_threshold) and \
                        (intersection_area > max_intersection_area):
                    max_intersection_area = intersection_area
                    field_to_increment = field_name
        if field_to_increment:
            filled_template[field_to_increment].append(word)
            matched_boxes[field_to_increment].append(ocr_box)

    filled_str_template = {
        key: " ".join(value) for key, value in filled_template.items()
    }
    return filled_str_template, matched_boxes


def main(cerfa_path: str):
    # Load Cerfa image
    # For plots
    image = load_cerfa(cerfa_path)
    width, height = image.size

    # OCR
    raw_ocr_output = ocrize(cerfa_path)
    ocr_output = process_doctr_output(raw_ocr_output, width, height)

    # Identify Cerfa using OCR output
    cerfa_number = identify_cerfa(raw_ocr_output)

    # Get template
    cerfa_template = get_cerfa_template(cerfa_number)
    clean_template = clean_cerfa_template(cerfa_template)

    # TODO: Clean up OCR output (untilting)
    # clean_ocr_output = clean_ocr_output(ocr_output)
    clean_ocr_output = ocr_output

    # Matching
    filled_template, matched_boxes = match_bounding_boxes_to_template(
        clean_ocr_output,
        clean_template,
        area_ratio_threshold=AREA_RATIO_THRESHOLD
    )
    print(filled_template)


if __name__ == "__main__":
    cerfa_path = sys.argv[1]
    main(cerfa_path=cerfa_path)
