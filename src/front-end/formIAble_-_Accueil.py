from PIL import Image
import numpy as np
import streamlit as st


# style CSS appliqué à la page HTML générée
pageStyleStr = """
<style>
    /* s'applique aux titres de type h1, dont ceux générés par st.title */
    h1 {
        text-align: center;
    }
    /*
       s'applique à la div contenant une image et dont l'attribut data-testid vaut "stImage".
       Malheureusement, aucun des paramètres de display ne fonctionne pour aligner l'image au centre car les paramètres
       de div d'un niveau supérieur s'appliquent en priorité.
    */
    /*
    div[data-testid="stImage"]>img {
        display: inline-block;
        display: table;
        display: flex;
        justify-content: center;
        height: fit-content;
        width: fit-content;
        width: intrinsic;
        max-width: 100%;
    }
    */
</style>
"""
st.write(pageStyleStr, unsafe_allow_html=True)

image = Image.open("./src/front-end/resources/formIAble_logo.png")
# crée 3 colonnes dont celle du centre (d'indice 1) qui doit occuper 90% de l'espace.
# Il s'agit de la seule façon probante de centrer l'image à l'écran avec Streamlit
with st.columns([1, 18, 1])[1]:
    # affiche l'image dans l'application
    st.image(np.array(image))

st.title("Application d'analyse automatique de formulaires CERFA")
