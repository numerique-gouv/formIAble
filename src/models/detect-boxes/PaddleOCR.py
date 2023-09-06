from typing import Dict, List, Union, Tuple
import json
from PIL import Image, ImageDraw
import logging
logging.basicConfig(level=logging.DEBUG)


def get_boxes_from_config_file(path_fields_file: str):
    with open(path_fields_file, "r") as fields_file:
        fields_dict: Dict = json.load(fields_file)
    fields_positions: List[List[int]] = []
    fields_names: List[str] = []
    for fields_dict_of_subtype in fields_dict.values():
        for field_name, field_type_and_pos in fields_dict_of_subtype.items():
            fields_positions.append(field_type_and_pos["pos"])
            fields_names.append(field_name)
    return fields_names, fields_positions


def plot_boxes(path_image_source: str,
               path_image_destination: str,
               positions: List[Union[List[int], List[float], List[List[int]]]]):
    for positions_of_box in positions:
        positions_to_plot: List[Tuple[float]] = []
        if isinstance(positions_of_box[0], int) or isinstance(positions_of_box[0], float):
            # positions_of_box is [x0, y0, x1, y1]
            x0, y0, x1, y1 = map(float, positions_of_box)
            positions_to_plot = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        elif isinstance(positions_of_box[0], list):
            assert len(positions_of_box) == 4, "Expected list of 4 corners in consecutive order"
            for coord_corner in positions_of_box:
                x, y = coord_corner
                positions_to_plot.append((x, y))
        else:
            raise ValueError("Input type must be List[Union[List[int], List[float], List[List[int]]]]")


get_boxes_from_config_file(path_fields_file="data/elements_to_fill_forms/non-editable/cerfa_14011_03_id.json")
