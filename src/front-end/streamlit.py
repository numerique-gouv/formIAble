from donut import DonutModel
from PIL import Image
from tempfile import NamedTemporaryFile
import json
import numpy as np
import os
import streamlit as st
import torch


# copie de la fonction de src/testing_donut.py car problème (de chemin) d'accès au module src/donut_lib dans l'application web
def run_model_on_file(model_path, file_path):
    with open(os.path.join(model_path, 'special_tokens_map.json'), "r") as file:
        special_tokens = json.load(file)
        prompt = special_tokens["additional_special_tokens"][0]
    model = DonutModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
    if torch.cuda.is_available():
        model.half()
        device = torch.device("cuda")
        model.to(device)
    else:
        model.to("cpu")
    model.eval()
    image = Image.open(file_path).convert("RGB")
    with torch.no_grad():
        output = model.inference(image=image, prompt=prompt)
        return output["predictions"][0]


st.title("Analyse automatique de formulaires CERFA")

@st.cache_data
def load_data():
    return

#data_load_state = st.text("Chargement des données...")
#data_load_state.text("")

st.subheader("Téléversement d'un formulaire au format JPG pour analyse")
st.write("Actuellement, seuls sont traités les CERFA 12485, 13479 et 14011.")
uploadedFile = st.file_uploader("Téléversez votre fichier JPG", type="jpg")
if uploadedFile is not None:
    image = Image.open(uploadedFile)
    st.write("Image téléversée avec succès")
    st.subheader("Affichage du formulaire téléversé")
    # affiche l'image dans l'application
    st.image(np.array(image))
    with NamedTemporaryFile(dir=".", suffix=".jpg") as withPathTemporaryUploadedFile:
        # chemin relatif du modèle DonUT
        modelPathStr = os.path.join("./src/models/donut_trained/20231002_095949")
        # sauvegarde temporairement sur le service le fichier téléversé
        withPathTemporaryUploadedFile.write(uploadedFile.getbuffer())
        # chemin absolu du fichier téléversé, stocké temporairerement, de la forme /home/onyxia/work/formIAble/tmpXXXXX.jpg
        uploadedFileFullPathStr = withPathTemporaryUploadedFile.name
        fieldsNamesAndValuesStrs = run_model_on_file(modelPathStr, uploadedFileFullPathStr)
        st.subheader("Résultat de l'analyse du formulaire téléversé")
        st.write(f"Couples \"nom du champ : valeur du champ\" lus dans le fichier {uploadedFile.name}")
        # affiche les couples clefs-valeurs reconnus
        for fieldNameStr, fieldValueStr in fieldsNamesAndValuesStrs.items():
            st.write(f"* {fieldNameStr} : {fieldValueStr}")
#else:
#    st.write("Merci de téléverser une image au format JPG uniquement !")
