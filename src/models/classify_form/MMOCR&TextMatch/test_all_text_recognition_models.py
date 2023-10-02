from typing import List
import os
from mmocr.apis import MMOCRInferencer
import logging
logging.basicConfig(level=logging.INFO)

# models for text detection
rec_models: List[str] = ["ABINet",
                         "SATRN",
                         "ABINet_Vision",
                         "MASTER",
                         "SAR",
                         "SATRN_sm",
                         "NRTR",
                         "RobustScanner",
                         "SDMGR"]

det_model: str = "FCE_IC15"
input_document_path: str = "data/synthetic_forms/cerfa_12485_03_fake1.jpg"
output_dir: str = "results/MMOCR/text_recognition"

logging.info(f"working directory = {os.getcwd()}")
for rec_model in rec_models:
    logging.info(f"Loading text recognition model {rec_model} ...")
    ocr = MMOCRInferencer(det=det_model, rec=rec_model)
    logging.info(f"Loading text recognition model {rec_model} [OK]")

    logging.info(f"Running text recognition model {rec_model} ...")
    output = ocr(
        input_document_path,
        out_dir=os.path.join(output_dir, rec_model),
        save_vis=True,
        save_pred=True
    )
    logging.info(f"Running text detection model {rec_model} [OK]")
