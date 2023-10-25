from PIL import Image
# importe le module de PyMuPDF permettant d'afficher et de manipuler par divers outils des documents PDF via Python
import fitz
from tempfile import NamedTemporaryFile
import numpy as np
# importe le module OS pour l'accès aux fonctions de gestion de fichiers et de chemins
import os
# importe le module de mesure du temps courant
import time
import streamlit as st

# importe le module des classes représentant le système de fichiers avec la sémantique appropriée pour différents systèmes
# d'exploitation (chemins orientés objet)
from pathlib import Path
# se positionne sur le chemin de travail courant
cwd = Path().resolve()
# importe le module des paramètres et fonctions systèmes
import sys
# ajoute le chemin de travail courant à la variable concaténant les répertoires système afin de permettre l'import
# de modules présents dans les sous-répertoires dudit répertoire
sys.path.append(str(cwd))
# importe le module de fonctions utiles
import src.util.utils as utilfunc
# importe le module d'exécution du modèle DonUT
import src.testing_donut as dot


st.title("Analyse automatique de formulaires CERFA")

@st.cache_data
def load_data():
    return

#data_load_state = st.text("Chargement des données...")
#data_load_state.text("")

st.subheader("Téléversement d'un formulaire au format JPG ou PDF pour analyse")
st.write("Actuellement, seuls sont traités les CERFA 12485, 13479 et 14011.")
uploadedFile = st.file_uploader("Téléversez votre fichier au format JPG ou PDF", type=["jpg", "pdf"])
if uploadedFile is not None:
    uploadedFileWithoutExtensionPathStr, uploadedFileExtensionStr = os.path.splitext(uploadedFile.name)
    if uploadedFileExtensionStr.lower() == ".pdf":
        lPdfDocument = fitz.open(stream=uploadedFile.read(), filetype="pdf")
        image = utilfunc.get_image_from_pdf_document(lPdfDocument)
    elif uploadedFileExtensionStr.lower() == ".jpg":
        image = Image.open(uploadedFile)
    uploadedFileAsImageRelativePathStr = f"./{uploadedFileWithoutExtensionPathStr}.jpg"
    image.save(fp=uploadedFileAsImageRelativePathStr, format="JPEG")
#    st.write(f"Document {uploadedFile.name} téléversé avec succès en tant que {uploadedFileAsImageRelativePathStr}")
    st.write(f"Document téléversé avec succès")
    st.subheader("Affichage du formulaire téléversé")
    # affiche l'image dans l'application
    st.image(np.array(image))
    with st.spinner(text="Analyse du document en cours..."):
        lBeforeProcessTime = time.time()
        # chemin relatif du modèle DonUT
        modelPathStr = "./data/models/donut_trained/20231002_095949"
#        with NamedTemporaryFile(dir=".", suffix=".jpg") as withPathTemporaryUploadedFile:
#            # sauvegarde temporairement le fichier téléversé sur le conteneur exécutant ce code
#            withPathTemporaryUploadedFile.write(uploadedFile.getbuffer())
#            # chemin absolu du fichier téléversé, stocké temporairerement, de la forme /home/onyxia/work/formIAble/tmpXXXXX.jpg
#            uploadedFileFullPathStr = withPathTemporaryUploadedFile.name
#            fieldsNamesAndValuesStrs = dot.run_model_on_file(modelPathStr, uploadedFileFullPathStr)
        fieldsNamesAndValuesStrs = dot.run_model_on_file(modelPathStr, uploadedFileAsImageRelativePathStr)
        st.subheader("Résultat de l'analyse du formulaire téléversé")
        st.write(f"""L'analyse du document {uploadedFile.name} s'est déroulée en {round(time.time() - lBeforeProcessTime, 2)} secondes
            et a pu extraire **{len(fieldsNamesAndValuesStrs)}** couples \"**nom du champ** : valeur du champ\" :""")
        # affiche les couples clefs-valeurs reconnus par DonUT et triés alphabétiquement par clef
        for fieldNameStr, fieldValueStr in sorted(fieldsNamesAndValuesStrs.items()):
            st.write(f"* **{fieldNameStr}** : {fieldValueStr}")
    # supprime l'image téléversée
    os.remove(uploadedFileAsImageRelativePathStr)
#else:
#    st.write("Merci de téléverser une image au format JPG uniquement !")
