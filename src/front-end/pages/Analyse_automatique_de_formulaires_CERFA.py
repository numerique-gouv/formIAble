from PIL import Image
from tempfile import NamedTemporaryFile
import numpy as np
import os
import streamlit as st

# importe le module des classes représentant le système de fichiers avec la sémantique appropriée pour différents systèmes d'exploitation
# (chemins orientés objet)
from pathlib import Path
# se positionne sur le chemin de travail courant
cwd = Path().resolve()
# importe le module des paramètres et fonctions systèmes
import sys
# ajoute le chemin de travail courant à la variable concaténant les répertoires système afin de permettre l'import
# de modules présents dans les sous-répertoires dudit répertoire
sys.path.append(str(cwd))
# importe le module d'exécution du modèle DonUT
import src.testing_donut as dot


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
    with st.spinner(text="Analyse du document en cours..."):
        with NamedTemporaryFile(dir=".", suffix=".jpg") as withPathTemporaryUploadedFile:
            # chemin relatif du modèle DonUT
            modelPathStr = os.path.join("./data/models/donut_trained/20231002_095949")
            # sauvegarde temporairement sur le service le fichier téléversé
            withPathTemporaryUploadedFile.write(uploadedFile.getbuffer())
            # chemin absolu du fichier téléversé, stocké temporairerement, de la forme /home/onyxia/work/formIAble/tmpXXXXX.jpg
            uploadedFileFullPathStr = withPathTemporaryUploadedFile.name
            fieldsNamesAndValuesStrs = dot.run_model_on_file(modelPathStr, uploadedFileFullPathStr)
            st.subheader("Résultat de l'analyse du formulaire téléversé")
            st.write(f"Couples \"**nom du champ** : valeur du champ\" lus dans le fichier {uploadedFile.name}")
            # affiche les couples clefs-valeurs reconnus par DonUT et triés alphabétiquement par clef
            for fieldNameStr, fieldValueStr in sorted(fieldsNamesAndValuesStrs.items()):
                st.write(f"* **{fieldNameStr}** : {fieldValueStr}")
#else:
#    st.write("Merci de téléverser une image au format JPG uniquement !")
