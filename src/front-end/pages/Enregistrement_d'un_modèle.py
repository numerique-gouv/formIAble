import streamlit as st
import time
import fitz
import PIL


def train_model():
    with st.empty():
        progress_text = "Ré-entraînement en cours."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()
        st.write("Modèle ré-entraîné et mis à jour !")
    return


def get_text_info(field):
    name = field.field_name.lower()
    rect = field.rect
    return name, rect


def get_radio_info(field):
    raise NotImplementedError()


def get_checkbox_info(field):
    raise NotImplementedError()


def get_page_template(field):
    fields_info = {
        "text": [],
        "radio": [],
        "checkbox": []
    }
    while field:
        if field.field_type == 7:
            fields_info["text"].append(get_text_info(field))
        elif field.field_type == 5:
            fields_info["radio"].append(get_radio_info(field))
        elif field.field_type == 2:
            fields_info["checkbox"].append(get_checkbox_info(field))
        field = field.next
    return fields_info


st.set_page_config(page_title="Enregistrement d'un nouveau formulaire")

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
    image: PIL.Image.Image = PIL.Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )

    # TODO: extraction de la position des champs pour display
    # boîtes
    field = page.first_widget

    st.image(image)

if st.button("Ré-entraînement"):
    train_model()
