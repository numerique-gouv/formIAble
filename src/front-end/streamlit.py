import streamlit as st
import pandas as pd
import numpy as np
import src.util.S3_storage as s3s


st.title("Analyse automatique de formulaires CERFA")

st.subheader("Téléversement d'un formulaire au format PDF pour analyse")
st.write("Attention !\nActuellement, seuls sont traités les CERFA ...")
uploaded_file = st.file_uploader("Téléversez votre fichier PDF", type="pdf")
if uploaded_file is not None:
    st.write(uploaded_file.name)

@st.cache_data
def load_data():
    return

data_load_state = st.text('Loading data...')
data_load_state.text("Done! (using st.cache_data)")
