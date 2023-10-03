from typing import Dict, List, Union, Tuple
import json
import PIL.Image
import PIL.ImageDraw
import logging
logging.basicConfig(level=logging.INFO)


def get_boxes_from_config_file(path_fields_file: str) -> Tuple[List[str], List[List[int]]]:
    """Read configuration file and load field names and positions"""
    with open(path_fields_file, "r") as fields_file:
        fields_dict: Dict = json.load(fields_file)
    fields_positions: List[List[int]] = []
    fields_names: List[str] = []
    for fields_dict_of_subtype in fields_dict.values():
        for field_name, field_type_and_pos in fields_dict_of_subtype.items():
            field_pos = field_type_and_pos["pos"]
            if isinstance(field_pos[0], list):
                for field_pos_ in field_pos:
                    fields_positions.append(field_pos_)
                    fields_names.append(field_name)
            else:
                fields_positions.append(field_pos)
                fields_names.append(field_name)
    return fields_names, fields_positions


def plot_boxes(path_image_source: str,
               path_image_destination: str,
               positions: List[Union[List[int], List[float], List[List[int]]]]):
    """Plot all boxes defined by their positions"""
    with PIL.Image.open(path_image_source, "r") as image_source:
        # convert to RGB to enable saving as jpeg
        image_destination: PIL.Image.Image = image_source.convert("RGB")
    draw = PIL.ImageDraw.Draw(image_destination)
    for i, positions_of_box in enumerate(positions):
        positions_to_plot: List[Tuple[float, float]] = []
        if isinstance(positions_of_box[0], int) or isinstance(positions_of_box[0], float):
            # positions_of_box is [x0, y0, x1, y1]
            x0, y0, width, height = map(float, positions_of_box)
            x1 = x0 + width
            y1 = y0 + height
            positions_to_plot = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        elif isinstance(positions_of_box[0], list):
            assert len(positions_of_box) == 4, "Expected list of 4 corners in consecutive order"
            for coord_corner in positions_of_box:
                x, y = coord_corner
                positions_to_plot.append((x, y))
        else:
            raise ValueError("Input type must be List[Union[List[int], List[float], List[List[int]]]]")
        draw.line(xy=positions_to_plot, fill="red")
    image_destination.save(path_image_destination)


fields_names, fields_positions = get_boxes_from_config_file("data/elements_to_fill_forms/non-editable/cerfa_14011_03_id.json")


plot_boxes(path_image_source="data/empty_forms/non-editable/cerfa_14011_03.png",
           path_image_destination="results/cerfa_14011_03.png",
           positions=fields_positions)
