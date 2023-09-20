from pdf2image import convert_from_path
import os
from pathlib import Path
import fitz  # package PyMuPDF
import PIL.Image


def ajout_retour_ligne(text, max_length, font, draw):

    new_text = text
    splitted_text = text.split(" ")
    i_start = 0
    for i in range(1, len(splitted_text) + 1):
        if "\n" in splitted_text[i-1]:
            i_start = i
        _, _, size, _ = draw.multiline_textbbox((0, 0), " ".join(splitted_text[i_start:i]), font=font)
        if size > max_length:
            new_text = " ".join(splitted_text[:i-1]) + "\n" + " ".join(splitted_text[i-1:])
            new_text = ajout_retour_ligne(new_text, max_length, font, draw)
    return new_text


def get_root_path() -> Path:
    return Path(__file__).parent.parent.parent


def convert_to_png(path_document_input: str,
                   path_document_output: str) -> None:
    """
    Export input document to png and save to path_document_output.
    In case of a pdf document with several pages, only the first one is exported.

    :param path_document_input: path of the document to export to png, extension can be pdf or a valid image extension
    :param path_document_output: path to save the exported document, extension must be .png
    """
    assert os.path.splitext(path_document_output)[1] == ".png", \
        f"Output path must have extension .png but path_document_output is {path_document_output}"
    extension_input: str = os.path.splitext(path_document_input)[1]
    if extension_input == ".pdf":
        doc = fitz.open(path_document_input)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=350)
        image: PIL.Image.Image = PIL.Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        image.save(fp=path_document_output, format="PNG")
    else:
        with PIL.Image.open(path_document_input) as image_input:
            image_input.save(fp=path_document_output, format="PNG")
