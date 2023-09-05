from typing import Dict, List
import os.path
import json
import paddleocr
import logging
logging.basicConfig(level=logging.DEBUG)


def plot_boxes_in_document(path_fields_file: str):
    with open(path_fields_file, "r") as fields_file:
        fields_dict: Dict = json.load(fields_file)
    fields_positions: List[List[int]] = []
    fields_names: List[str] = []
    for fields_dict_of_subtype in fields_dict.values():
        for field_name, field_type_and_pos in fields_dict_of_subtype.items():
            fields_positions.append(field_type_and_pos["pos"])
            fields_names.append(field_name)
    logging.debug(fields_positions)
    logging.debug(fields_names)


plot_boxes_in_document(path_fields_file="data/elements_to_fill_forms/non-editable/cerfa_14011_03_id.json")


