import logging
import paddleocr
import src.pipeline.pipeline_PaddleOCR
logging.basicConfig(level=logging.DEBUG)


ocr_model: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=True,
                                                     lang='fr')


src.pipeline.pipeline_PaddleOCR.extract_document(path_document_input="data/synthetic_forms/cerfa_12485_03_fake1.jpg",
                                                 path_dir_configs="data/configs_extraction",
                                                 ocr_model=ocr_model)

# src.pipeline.pipeline_PaddleOCR.register_document(path_document_to_register="data/empty_forms/non-editable/cerfa_12485_03.png",
#                                                   path_dir_reference_documents="data/configs_extraction",
#                                                   texts_reference=["cerfa", "S 3704b", "Déclaration signée le"],
#                                                   ocr_model=ocr_model)
