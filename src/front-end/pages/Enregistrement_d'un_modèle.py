from pathlib import Path
import sys

cwd = Path().resolve()
sys.path.append(str(cwd))

from src.util.dataGeneration import Writer13753_04
from src.util.dataGeneration.baseWriter import AnnotatorTxt
from typing import Dict, List, Tuple
import streamlit as st
import time
import fitz
import random
from PIL import ImageDraw, Image, ImageFont, ImageFilter


def train_model():
    """
    Dummy train model fn.
    """
    with st.empty():
        progress_text = "Ré-entraînement en cours."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()
        st.write("Modèle ré-entraîné et mis à jour !")
    return


@st.cache_data
def generate_data(n: int) -> List[Image.Image]:
    """Generate n examples of Cerfa 13753_04.

    Args:
        n (int): Number of examples to generate

    Returns:
        List[Image.Image]: List of images.
    """
    # Only generates 13753_04, to adjust
    num_cerfa = "13753_04"
    images = []

    for _ in range(n):
        annotator = AnnotatorTxt()
        writer = Writer13753_04(num_cerfa=num_cerfa, annotator=annotator)
        writer.fill_form()
        pix = writer.doc.load_page(0).get_pixmap(dpi=350)
        image: Image.Image = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples
        )
        images.append(image)
    return images


def get_text_info(field) -> Tuple:
    """
    Get name and coordinates of text field in PDF.
    """
    # Coordinates of rect: x0, y0, x1, y1.
    # They are treated as being coordinates of two diagonally
    # opposite points. The first two numbers are regarded as the
    # top left corner and x1, y1 as the bottom right one.
    name = field.field_name.lower()
    rect = field.rect
    return name, rect


def get_radio_info(field):
    """
    Get name and coordinates of radio field in PDF.
    """
    raise NotImplementedError()


def get_checkbox_info(field):
    """
    Get name and coordinates of checkbox field in PDF.
    """
    raise NotImplementedError()


def get_page_template(field) -> Dict:
    """
    Get template of Cerfa PDF page.
    """
    fields_info = {
        "text": [],
        "radio": [],
        "checkbox": []
    }
    while field:
        if field.field_type == 7:
            fields_info["text"].append(get_text_info(field))
        # elif field.field_type == 5:
        #     fields_info["radio"].append(get_radio_info(field))
        # elif field.field_type == 2:
        #     fields_info["checkbox"].append(get_checkbox_info(field))
        field = field.next
    return fields_info


def update_chosen_types_from_selection(chosen_types: Dict, i: int):
    """
    Update the dictionary of field types for field i according
    to what is specified by the user.

    Args:
        chosen_types (Dict): Dictionary of field types.
        i (int): Field id.
    """
    chosen_types[i]["type"] = st.selectbox(
        f"Type du champ {i}",
        field_types,
        key=i,
        index=default_indices[i]["index"]
    )
    placeholder = st.empty()
    with placeholder.container():
        st.text("")
    if chosen_types[i]["type"] == "Nombre dans un intervalle":
        min_value = default_indices[i]["kwargs"].get("min_value", None)
        max_value = default_indices[i]["kwargs"].get("max_value", None)
        with placeholder.container():
            container_col1, container_col2 = st.columns(2)
            with container_col1:
                min_value = st.text_input("Valeur minimum", value=min_value, key=f"{i}_1")
            with container_col2:
                max_value = st.text_input("Valeur maximum", value=max_value, key=f"{i}_2")
        chosen_types[i]["kwargs"]["interval"] = (int(min_value), int(max_value))
    return


@st.cache_data
def tilt_image(image: Image.Image) -> Image.Image:
    """
    Tilt image by a random amount (small angle) for
    demo purposes.

    Args:
        image (Image.Image): Image.

    Returns:
        Image.Image: Tilted image.
    """
    angle = random.uniform(-10.0, -5.0)
    if random.randint(0, 1) == 0:
        angle = random.uniform(5.0, 10.0)
    return image.rotate(angle, Image.NEAREST, expand=1, fillcolor="white")


@st.cache_data
def blur_image(image: Image.Image) -> Image.Image:
    """
    Blur image for demo purposes.

    Args:
        image (Image.Image): Image.

    Returns:
        Image.Image: Blurred image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=2))


def click_generate_button():
    st.session_state.generate_clicked = True


# Streamlit App
st.set_page_config(page_title="Enregistrement d'un nouveau formulaire")


# Custom CSS to change background color of text input boxes
st.markdown(
    """
    <style>
    .stTextInput input[type="text"] {
        background-color: #a9cfbf;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("# Enregistrement d'un nouveau formulaire")
st.sidebar.header("Enregistrement")
st.write(
    """Ici il est possible d'ajouter un nouveau formulaire à l'outil
    d'extraction."""
)

# Upload PDF
uploaded_pdf = st.file_uploader(
    "Téléversez votre formulaire Cerfa "
    "au format PDF (éditable)",
    type="pdf"
)

if uploaded_pdf is not None:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=350)
    image: Image.Image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )
    image_width, image_height = image.size

    # Coordinates of fields
    field = page.first_widget
    page_template = get_page_template(field)

    image_bb = ImageDraw.Draw(image)
    font = ImageFont.truetype("data/arial.ttf", 40)
    for i, text_field in enumerate(page_template["text"]):
        x0, y0, x1, y1 = text_field[1]

        # Cast coordinates to image coordinate system
        x0 = int(x0 * image_width / page.rect.width)
        x1 = int(x1 * image_width / page.rect.width)
        y0 = int(y0 * image_height / page.rect.height)
        y1 = int(y1 * image_height / page.rect.height)

        # Plot bounding boxes
        image_bb.rectangle((x0, y0, x1, y1), fill="#f0f2f6")
        # Text coordinates
        x = int((x0 + x1) / 2)
        y = int((y0 + y1) / 2)
        # Write text
        image_bb.text((x, y), text=str(i), fill="#000000", font=font)

    st.image(image)

    # Field types
    field_types = (
        "Nom",
        "Nombre dans un intervalle",
        "Date",
        "Ville",
        "Code postal",
        "Paragraphe (texte)",
        "Numéro de voie",
        "Extension d'adresse",
        "Type de voie",
        "Nom de rue",
        "Plaque d'immatriculation",
        "Marque de voiture",
        "Numéro SIRET"
    )
    default_indices = {
        0: {"index": 0, "kwargs": {}},
        1: {"index": 12, "kwargs": {}},
        2: {"index": 3, "kwargs": {}},
        3: {"index": 6, "kwargs": {}},
        4: {"index": 7, "kwargs": {}},
        5: {"index": 8, "kwargs": {}},
        6: {"index": 9, "kwargs": {}},
        7: {"index": 4, "kwargs": {}},
        8: {"index": 3, "kwargs": {}},
        9: {"index": 1, "kwargs": {"min_value": "1", "max_value": "31"}},
        10: {"index": 1, "kwargs": {"min_value": "1", "max_value": "12"}},
        11: {"index": 1, "kwargs": {"min_value": "2000", "max_value": "2023"}},
        12: {"index": 10, "kwargs": {}},
        13: {"index": 11, "kwargs": {}},
        14: {"index": 11, "kwargs": {}},
        15: {"index": 2, "kwargs": {}},
        16: {"index": 3, "kwargs": {}},
        17: {"index": 5, "kwargs": {}},
        18: {"index": 3, "kwargs": {}},
        19: {"index": 2, "kwargs": {}},
    }
    chosen_types = {
        i: {"type": None, "kwargs": {}}
        for i in range(len(page_template["text"]))
    }
    col1, col2, col3 = st.columns(3)
    # Loop through questions and display them in columns accordingly
    for i, _ in enumerate(page_template["text"]):
        if i % 3 == 0:
            with col1:
                update_chosen_types_from_selection(chosen_types, i)
        elif i % 3 == 1:
            with col2:
                update_chosen_types_from_selection(chosen_types, i)
        else:
            with col3:
                update_chosen_types_from_selection(chosen_types, i)

    if 'generate_clicked' not in st.session_state:
        st.session_state.generate_clicked = False
    st.button(
        "Générer des données d'entraînement",
        on_click=click_generate_button
    )
    if st.session_state.generate_clicked:
        # Generate data
        # Here we generate with the already existing Writer class and
        # config .json but need to dynamically create them / simplify
        # the code architecture to avoid creating a new class everytime ?
        image = generate_data(n=1)[0]

        # Augment data
        tilted_image = tilt_image(image)
        blurred_image = blur_image(image)

        # Display examples
        image_col, tilted_col, blurred_col = st.columns(3)

        with image_col:
            st.image(image)
        with tilted_col:
            st.image(tilted_image)
        with blurred_col:
            st.image(blurred_image)

    if st.button("Ré-entraînement"):
        train_model()
