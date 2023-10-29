# importe le module de génération de fichiers et de répertoires temporaires, dont des fichiers temporaires nommés
#from tempfile import NamedTemporaryFile
# importe le module de traitement scientifique, dont les tableaux multi-dimensionnels
import numpy as np
# importe le module OS pour l'accès aux fonctions de gestion de fichiers et de chemins
import os
# importe le module de mesure du temps courant
import time
# importe le module de gestion des images (Python Imaging Library)
from PIL import Image
# importe le module de boîtes à outils d'OCRisation basées sur PaddlePaddle (PArallel Distributed Deep LEarning),
# la seule plateforme chinoise indépendante de deep learning pour la R&D, ouverte à la communauté open source
# depuis 2016
import paddleocr as ocr
# importe le module de création et du publication d'applications basées sur des données
import streamlit as st

# importe le module des classes représentant le système de fichiers avec la sémantique appropriée pour différents
# systèmes d'exploitation (chemins orientés objet)
from pathlib import Path
# importe le module des paramètres et fonctions systèmes
import sys
# se positionne sur le chemin de travail courant
cwd = Path().resolve()
# ajoute le chemin de travail courant à la variable concaténant les répertoires système afin de permettre l'import
# de modules présents dans les sous-répertoires dudit répertoire
sys.path.append(str(cwd))
# importe le module de constantes et fonctions utiles
import src.util.utils as utils
# importe le module de PaddleOCR permettant l'extraction d'informations et la classification d'un fichier PDF selon
# l'un des modèles connus
import src.models.classify_form.PaddleOCR_TextMatch.classify as ocrExtractor
# importe le module des étapes d'exécution de PaddleOCR
import src.pipeline.pipeline_PaddleOCR as ocrPipeline
# importe le module d'exécution du modèle DonUT
import src.testing_donut as dot


@st.cache_resource
def load_ocr_model():
    r"""Charge le modèle PaddleOCR en mémoire lors du premier chargement de la page.
    Le décorateur `@st.cache_resource` permet de mettre en cache le modèle sans avoir à le sérialiser comme avec le
    décorateur `@st.cache_data`.

    Returns
    -------
    paddleocr.paddleocr.PaddleOCR
        le modèle d'OCR PaddleOCR après son téléchargement.
    """
    return ocr.PaddleOCR(use_angle_cls=True, lang='fr')


data_load_state = st.text("Chargement des modèles...")
# charge le modèle PaddleOCR en mémoire lors du premier chargement de la page
paddleOcrModel = load_ocr_model()
data_load_state.text("")

st.title("Analyse automatique de formulaires CERFA")
st.subheader("Téléversement d'un formulaire au format JPG, PNG ou PDF pour analyse")
st.write("Actuellement, seuls sont traités les CERFA 12485\*03, 13749\*05 et 14011\*03.")
uploadedFile = st.file_uploader("Téléversez votre fichier au format JPG, PNG ou PDF", type=["jpg", "jpeg", "png", "pdf"])
# si un fichier correspondant aux extensions acceptées a été téléversé,
if uploadedFile is not None:
    # chemin du fichier téléversé lorsque sauvegardé localement
    uploadedFileRelativePathStr = f"{utils.TEMPORARY_FILE_DIRECTORY_STR}/{uploadedFile.name}"
    # sauvegarde localement le contenu binaire (tableau d'octets) du fichier téléversé
    with open(uploadedFileRelativePathStr, "wb") as savedUploadedFile:
        savedUploadedFile.write(uploadedFile.read())
    # extrait du nom dudit fichier :
    # - son chemin et la racine de son nom, sans extension
    # - son extension
    #uploadedFileWithoutExtensionPathStr, uploadedFileExtensionStr = os.path.splitext(uploadedFile.name)
    uploadedFileAsImage, uploadedFileAsImageRelativePathStr = utils.get_and_save_image_and_path_from_document(
        input_document_path=uploadedFileRelativePathStr,
        output_image_format="PNG",
        output_image_quality_in_dpi=350
    )
#    st.write(f"Document {uploadedFile.name} téléversé avec succès en tant que {uploadedFileAsImageRelativePathStr}")
    st.write(f"Document téléversé avec succès")
    st.subheader("Affichage du formulaire téléversé")
    # affiche l'image dans l'application
    st.image(np.array(uploadedFileAsImage))
    # affiche un cercle de chargement durant le processus d'extraction du numéro CERFA
    with st.spinner(text="Extraction du numéro CERFA du document en cours..."):
        lBeforeProcessTime = time.time()
        # liste des éléments de texte extraits et des coordonnées des points définissant les boîtes entourant lesdits éléments
        # de texte extraits de l'OCRisation de l'image png_document_path_str par le modèle d'OCR PaddleOCR
        lInputDocumentTextElements, lInputDocumentTextBoxes = \
            ocrPipeline.get_form_image_text_elements_and_boxes(uploadedFileAsImageRelativePathStr, paddleOcrModel)
        # extrait le numéro CERFA du formulaire afin de trouver le fichier de configuration dudit formulaire, définissant
        # les champs, leurs positions, les éléments de texte de référence et la taille de l'image de référence associée
        cerfaFormNumberStr: str = ocrExtractor.get_form_number_in_text_elements(
            input_document_path=uploadedFile.name,
            text_elements=lInputDocumentTextElements
        )
        st.subheader("Numéro CERFA du formulaire téléversé")
        st.write(f"""Le **numéro CERFA** du document {uploadedFile.name} est le **{cerfaFormNumberStr}**.
            Son extraction s'est déroulée en {round(time.time() - lBeforeProcessTime, 2)} secondes.""")
    # affiche un cercle de chargement durant le processus de prétraitement du formulaire CERFA
    with st.spinner(text="Prétraitement du document en cours..."):
        lBeforeProcessTime = time.time()
        try:
            # - trouve les boîtes entourant les éléments de texte correspondant le mieux aux éléments de texte de
            #   l'image de référence du formulaire
            # - applique une rotation à l'image, en sauvegarde le résultat dans un fichier différent pour éviter
            #   tout conflit de nommage et retourne l'image obtenue et la matrice de transformation de ladite image
            lTransformedImageRelativePathStr, lTransformationMatrix = ocrPipeline.get_transformationMatrix_and_save_image_after_affineTransformation(
                input_document_path_str=uploadedFile.name,
                form_image_path_str=uploadedFileAsImageRelativePathStr,
                form_number_str=cerfaFormNumberStr,
                input_document_text_elements=lInputDocumentTextElements,
                input_document_text_boxes=lInputDocumentTextBoxes,
                configuration_files_dir_path_str="./data/configs_extraction",
            )
        except AssertionError as lAssertionError:
            st.write(f"""Le prétraitement du document {uploadedFile.name} n'est pas possible :
                **aucun fichier de configuration correspondant** n'existe pour ce formulaire.""")
#            st.write(f"{lAssertionError}")
            lTransformedImageRelativePathStr = uploadedFileAsImageRelativePathStr
        else:
            st.subheader("Affichage du formulaire téléversé après prétraitement")
            lTransformedImage = Image.open(lTransformedImageRelativePathStr)
            # affiche l'image dans l'application
            st.image(np.array(lTransformedImage))
            st.write(f"""Le prétraitement du document {uploadedFile.name} s'est déroulé en
                {round(time.time() - lBeforeProcessTime, 2)} secondes.""")
    # affiche un cercle de chargement durant le processus d'analyse du formulaire CERFA
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
        # couples {nom du champ : valeur du champ} lus par le modèle d'OCR DonUT
        fieldsNamesAndValuesStrs = dot.run_model_on_file(modelPathStr, lTransformedImageRelativePathStr)
        st.subheader("Résultat de l'analyse du formulaire téléversé")
        st.write(f"""L'analyse du document {uploadedFile.name} s'est déroulée en {round(time.time() - lBeforeProcessTime, 2)} secondes
            et a pu extraire **{len(fieldsNamesAndValuesStrs)}** couples \"**nom du champ** : valeur du champ\" :""")
        # affiche les couples clefs-valeurs reconnus par DonUT et triés alphabétiquement par clef
        for fieldNameStr, fieldValueStr in sorted(fieldsNamesAndValuesStrs.items()):
            st.write(f"* **{fieldNameStr}** : {fieldValueStr}")
    # supprime l'image transformée obtenue après prétraitement de l'image créée à partir du document téléversé
    if lTransformedImageRelativePathStr not in [uploadedFileRelativePathStr, uploadedFileAsImageRelativePathStr]:
        os.remove(lTransformedImageRelativePathStr)
    # supprime l'image créée à partir du document téléversé
    os.remove(uploadedFileAsImageRelativePathStr)
    # supprime le document téléversé si ce n'est pas l'image créée
    if uploadedFileRelativePathStr not in [uploadedFileAsImageRelativePathStr, lTransformedImageRelativePathStr]:
        os.remove(uploadedFileRelativePathStr)
#else:
#    st.write("Merci de téléverser une image au format JPG uniquement !")
