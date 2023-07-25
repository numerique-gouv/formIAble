"""
"""
from pathlib import Path
from typing import List

from doctr.io import Document, DocumentFile
from doctr.models import ocr_predictor
import fitz
import re


def get_list_words_in_page(page: Document):
    list_words_in_page = []
    for block in page.blocks:
        for line in block.lines:
            list_words_in_page.extend(line.words)
    return list_words_in_page


class DoctrTransformer:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, raw_documents):
        doctr_documents = self._get_doctr_docs(raw_documents=raw_documents)
        return doctr_documents

    def _get_doctr_docs(self, raw_documents: List[Path]):
        if not hasattr(self, "doctr_model"):
            self.doctr_model = ocr_predictor(
                det_arch="db_resnet50",
                reco_arch="crnn_vgg16_bn",
                pretrained=True,
            )
        list_doctr_docs = []
        for doc in raw_documents:
            if not doc.exists():
                print(f"Doc {doc} could not be found.")
                continue
            res_doctr = None
            try:
                if doc.suffix == "pdf":
                    doc_doctr = DocumentFile.from_pdf(doc)
                else:
                    doc_doctr = DocumentFile.from_images(doc)
                res_doctr = self.doctr_model(doc_doctr)
            except Exception as e:
                print(f"Could not analyze document {doc}. Error: {e}")
            if res_doctr:
                list_doctr_docs.append(res_doctr)

        return list_doctr_docs


def identify_cerfa(page_content):
    pattern = r"N[°o](\d{5}\*\d{2})"

    match = re.search(pattern, page_content, re.IGNORECASE)

    if match:
        result = match.group(1)
        return result
    else:
        raise ValueError("No cerfa number found.")


def main(transformer, path):
    # Load document if pdf
    if Path(path).suffix == ".pdf":
        doc = fitz.open(path)

        # If text is available directly
        page = doc.load_page(0)
        page_content = page.get_text()
        if page_content:
            return identify_cerfa(page_content)

    doctr_documents = transformer.transform([Path(path)])
    page = doctr_documents[0].pages[
        0
    ]

    page_content = ""
    for block in page.blocks:
        for line in block.lines:
            content = " ".join([word.value for word in line.words])
            page_content += content + "\n"
    return identify_cerfa(page_content)


if __name__ == "__main__":
    transformer = DoctrTransformer()
    path_list = [
        "../data/synthetic_forms/cerfa_12485_03_fake1.jpg",
        "../data/empty_forms/non-editable/cerfa_12485_03.png",
        "../data/empty_forms/non-editable/cerfa_14011_03.png",
        "../data/empty_forms/editable/cerfa_13753_04.pdf",
        "../data/empty_forms/editable/cerfa_13969_01.pdf"
    ]

    for path in path_list:
        try:
            cerfa_number = main(transformer, path)
            print(cerfa_number)
        except Exception:
            pass
