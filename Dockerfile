ARG BASE_IMAGE=inseefrlab/onyxia-python-pytorch
FROM $BASE_IMAGE

WORKDIR /app

# Clone repository
RUN git clone https://github.com/etalab-ia/formIAble.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/front-end/formIAble_-_Accueil.py", "--server.port=8501", "--server.address=0.0.0.0"]