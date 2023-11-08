ARG BASE_IMAGE=inseefrlab/onyxia-python
FROM $BASE_IMAGE

USER root

WORKDIR /app

# DEPENDENCES PADDLEOCR
# màj les listes des référentiels de paquetages APT
RUN apt update
# installe des bibliothèques système
RUN apt-get install -y libgl1 libglib2.0-0
# installe cargo depuis cURL via un appel à sh avec les options :
# - -s, qui permet d'envoyer des options à une commande depuis sh
# - --, qui est ensuite nécessaire pour séparer les options de sh des options passées pour cURL
# - -y, qui permet d'activer le mode non interactif de cURL et de répondre automatiquement "yes" aux questions posées
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# màj OpenSSL pour faire fonctionner PaddlePaddle (nécessaire à PaddleOCR) correctement
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb
RUN dpkg -i libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb
RUN rm libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb

# Clone repository
RUN git clone https://github.com/etalab-ia/formIAble.git .

RUN pip3 install -r requirements.txt

# installe la boîte à outils MMOCR
RUN mim install mmengine
RUN mim install mmcv
RUN mim install mmdet
RUN mim install mmocr

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/front-end/formIAble_-_Accueil.py", "--server.port=8501", "--server.address=0.0.0.0"]