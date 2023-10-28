# importe des classes du module permettant la compatibilité avec les conseils sur le typage (type hints)
from typing import Final, List, Tuple
from pdf2image import convert_from_path
# importe le module OS pour l'accès aux fonctions de gestion de fichiers et de chemins
import os
# importe le module des classes représentant le système de fichiers avec la sémantique appropriée pour différents
# systèmes d'exploitation (chemins orientés objet)
from pathlib import Path
# importe le module de PyMuPDF permettant d'afficher et de manipuler par divers outils des documents PDF via Python
import fitz
import logging
from tqdm import tqdm
# importe le module de gestion des images (Python Imaging Library)
from PIL import Image


# --- Constantes ---
# répertoire des fichiers temporaires
TEMPORARY_FILE_DIRECTORY_STR: Final[str] = "./data/tmp"
# préfixe des fichiers temporaires
TEMPORARY_FILE_PREFIX_STR: Final[str] = "temp"
# liste des extensions d'image acceptées
VALID_OUTPUT_IMAGE_EXTENSIONS: Final[List[str]] = [".jpg", ".jpeg", ".png"]


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


def get_image_from_pdf_document(input_pdf_document: fitz.Document, image_quality_in_dpi: int = 350) -> Image:
    r"""Exporte le fichier PDF `input_pdf_document` accessible en tant que document PDF vers une Image
    avec une qualité de `image_quality_in_dpi` dpi.

    Parameters
    ----------
    input_pdf_document : fitz.Document
        le document PDF à ouvrir et à transformer en image.
    image_quality_in_dpi : int, default=350
        la qualité en dpi (dot per inch) de l'image à générer à partir du fichier PDF `input_pdf_document`.

    Returns
    -------
    Image
        l'image du fichier PDF `input_pdf_document` avec la qualité `image_quality_in_dpi` indiquée.
    """
    lPage = input_pdf_document.load_page(0)
    lPixelMap = lPage.get_pixmap(dpi=image_quality_in_dpi)
    lImage = Image.frombytes(
        "RGB",
        (lPixelMap.width, lPixelMap.height),
        lPixelMap.samples
    )
    return lImage


def get_and_save_image_from_document(
    input_document_path: str,
    output_image_path: str,
    output_image_format: str = "PNG",
    output_image_quality_in_dpi: int = 350
) -> Image:
    r"""Exporte le document en entrée appelé `input_document_path` vers une image au format `output_image_format`,
    la sauvegarde dans un fichier appelé `output_image_path` et en retourne le contenu.
    S'il s'agit d'un document PDF de plusieurs pages, seule la première page est exportée.

    Parameters
    ----------
    input_document_path : str
        le chemin du document à exporter en tant qu'image, son extension peut être ".pdf" ou une extension d'image valide.
    output_image_path : str
        le chemin de l'image en sortie résultant de la conversion du document en entrée.
    output_image_format : str, default="PNG"
        le format de l'image en sortie.
    output_image_quality_in_dpi : int, default=350
        la qualité en dpi (dot per inch) de l'image à générer à partir du fichier `input_document_path`.

    Returns
    -------
    Image
        l'image `output_image_path` du fichier `input_document_path` au format `output_image_format` et avec
        la qualité `output_image_quality_in_dpi` indiquée.
    """
    # extension de l'image en sortie, obtenue à partir du chemin de l'image en sortie
    output_image_extension_str = os.path.splitext(output_image_path)[1]
    # vérifie que l'extension de l'image en sortie est bien l'une des extensions acceptées
    assert output_image_extension_str.lower() in VALID_OUTPUT_IMAGE_EXTENSIONS, \
        f"Output path must have one of these extensions: {VALID_OUTPUT_IMAGE_EXTENSIONS}, but the output path ends with {output_image_extension_str}"
    # extension du document en entrée, obtenue à partir du chemin du document en entrée
    input_document_extension_str: str = os.path.splitext(input_document_path)[1]
    # si le fichier téléversé est un document PDF,
    if input_document_extension_str.lower() == ".pdf":
        lPdfDocument = fitz.open(input_document_path)
        # le transforme en image
        image = get_image_from_pdf_document(input_pdf_document=lPdfDocument, image_quality_in_dpi=output_image_quality_in_dpi)
    # si le fichier téléversé est une image,
    else:
        # en charge le contenu
        image = Image.open(input_document_path)
    # sauvegarde l'image obtenue dans le format passé en paramètre
    image.save(fp=output_image_path, format=output_image_format)
    return image


def get_and_save_image_and_path_from_document(
    input_document_path: str,
    output_image_format: str = "PNG",
    output_image_quality_in_dpi: int = 350
) -> Tuple[Image, str]:
    r"""Exporte le document (PDF ou image) appelé `input_document_path` en tant qu'image de format `output_image_format`,
    la sauvegarde dans un fichier, et retourne le contenu de l'image ainsi exportée et son chemin.
    S'il s'agit d'un document PDF de plusieurs pages, seule la première page est exportée.

    Parameters
    ----------
    input_document_path : str
        le chemin du document à exporter en tant qu'image, son extension peut être ".pdf" ou une extension d'image valide.
    output_image_format : str, default="PNG"
        le format de l'image en sortie.
    output_image_quality_in_dpi : int, default=350
        la qualité en dpi (dot per inch) de l'image à générer à partir du fichier `input_document_path`.

    Returns
    -------
    - Image
        l'image résultant de la conversion du fichier `input_document_path` en image au format `output_image_format` et
        avec la qualité `output_image_quality_in_dpi` indiquée.
    - str
        le chemin de l'image exportée à partir du fichier `input_document_path`.
    """
    # l'extension du fichier en sortie est le format mis en minuscule
    output_image_extension_str = output_image_format.lower()
    # si le format de sortie de l'image est JPEG,
    if output_image_format == "JPEG":
        # définit l'extension à jpg
        output_image_extension_str = "jpg"
    input_document_with_image_extension_str: str = \
        f"{os.path.splitext(os.path.basename(input_document_path))[0]}.{output_image_extension_str}"
    # chemin de l'image en sortie
    output_image_path_str: str = f"{TEMPORARY_FILE_DIRECTORY_STR}/{TEMPORARY_FILE_PREFIX_STR}_{input_document_with_image_extension_str}"
    logging.debug(f"Nom de l'image en sortie = {output_image_path_str}")
    # image en sortie obtenue après conversion du document en entrée en image
    output_image = get_and_save_image_from_document(
        input_document_path=input_document_path,
        output_image_path=output_image_path_str,
        output_image_format=output_image_format,
        output_image_quality_in_dpi=output_image_quality_in_dpi
    )
    return output_image, output_image_path_str


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
