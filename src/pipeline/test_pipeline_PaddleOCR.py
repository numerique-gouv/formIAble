import logging
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
# importe le module d'exécution de PaddleOCR
import src.pipeline.pipeline_PaddleOCR
logging.basicConfig(level=logging.DEBUG)


ocr_model: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                     lang='fr')


src.pipeline.pipeline_PaddleOCR.extract_document(path_document_input="./data/synthetic_forms/cerfa_12485_03_fake1.jpg",
                                                 path_dir_configs="./data/configs_extraction",
#                                                 path_dir_configs="./data/elements_to_fill_forms/non-editable",
                                                 ocr_model=ocr_model)

# src.pipeline.pipeline_PaddleOCR.register_document(path_document_to_register="data/empty_forms/non-editable/cerfa_12485_03.png",
#                                                   path_dir_reference_documents="data/configs_extraction",
#                                                   texts_reference=["cerfa", "S 3704b", "Déclaration signée le"],
#                                                   ocr_model=ocr_model)
