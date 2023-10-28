import logging
# importe le module de boîtes à outils d'OCRisation basées sur PaddlePaddle (PArallel Distributed Deep LEarning),
# la seule plateforme chinoise indépendante de deep learning pour la R&D, ouverte à la communauté open source
# depuis 2016
import paddleocr

# importe le module des classes représentant le système de fichiers avec la sémantique appropriée pour différents systèmes d'exploitation
# (chemins orientés objet)
from pathlib import Path
# importe le module des paramètres et fonctions systèmes
import sys
# se positionne sur le chemin de travail courant
cwd = Path().resolve()
# ajoute le chemin de travail courant à la variable concaténant les répertoires système afin de permettre l'import
# de modules présents dans les sous-répertoires dudit répertoire
sys.path.append(str(cwd))
# importe le module des étapes d'exécution de PaddleOCR
import src.pipeline.pipeline_PaddleOCR as ocrPipeline
logging.basicConfig(level=logging.DEBUG)


ocr_model: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                     lang='fr')


ocrPipeline.extract_document(
    input_document_path="./data/synthetic_forms/cerfa_12485_03_fake1.jpg",
    configuration_files_dir_path="./data/configs_extraction",
#   configuration_files_dir_path="./data/elements_to_fill_forms/non-editable",
    ocr_model=ocr_model
)

# ocrPipeline.register_document(
#     document_to_register_path="data/empty_forms/non-editable/cerfa_12485_03.png",
#     reference_documents_dir_path="data/configs_extraction",
#     document_to_register_reference_texts=["cerfa", "S 3704b", "Déclaration signée le"],
#     ocr_model=ocr_model
# )
