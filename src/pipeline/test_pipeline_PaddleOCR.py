import paddleocr
import src.pipeline.pipeline_PaddleOCR

ocr_model: paddleocr.PaddleOCR = paddleocr.PaddleOCR(use_angle_cls=False,
                                                     lang='fr')

src.pipeline.pipeline_PaddleOCR.register_document(path_document_to_register="data/empty_forms/non-editable/cerfa_12485_03.png",
                                                  path_dir_reference_documents="data/configs_extraction",
                                                  texts_reference=["cerfa", "S 3704b", "Déclaration signée le"])
