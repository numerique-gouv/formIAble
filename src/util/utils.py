from pdf2image import convert_from_path
import os
from pathlib import Path
# importe le module de PyMuPDF permettant d'afficher et de manipuler par divers outils des documents PDF via Python
import fitz
import PIL.Image
from tqdm import tqdm
from PIL import Image


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


def get_image_from_pdf_document(pdf_document: fitz.Document, image_quality_in_dpi: int = 350) -> Image:
    r"""Retourne le fichier PDF chargé accessible en tant que document `pdf_document` au format Image
    avec une qualité de `image_quality_in_dpi` dpi.

    Parameters
    ----------
    pdf_document : fitz.Document
        le document PDF à ouvrir et à transformer en image.
    image_quality_in_dpi: int, default=350
        la qualité en dpi (dot per inch) de l'image à générer à partir du fichier PDF `pdf_document`.

    Returns
    -------
    Image
        l'image du fichier PDF `pdf_document` avec la qualité `image_quality_in_dpi` indiquée.
    """
    lPage = pdf_document.load_page(0)
    lPixelMap = lPage.get_pixmap(dpi=image_quality_in_dpi)
    lImage = Image.frombytes(
        "RGB",
        (lPixelMap.width, lPixelMap.height),
        lPixelMap.samples
    )
    return lImage


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
        image = get_image_from_pdf_document(pdf_document=doc, image_quality_in_dpi=350)
        image.save(fp=path_document_output, format="PNG")
    else:
        with PIL.Image.open(path_document_input) as image_input:
            image_input.save(fp=path_document_output, format="PNG")


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def pdf_to_image(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(".pdf"):
            # print(f"Processing {filename}")
            pdf_path = os.path.join(input_folder, filename)
            images = convert_from_path(pdf_path)
            image = images[0]
            for next_image in images[1:]:
                image = get_concat_h(image, next_image)
            output_filename = f"{filename[:-4]}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            image.save(output_path)
