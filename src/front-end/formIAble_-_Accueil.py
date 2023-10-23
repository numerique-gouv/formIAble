from PIL import Image
import numpy as np
import streamlit as st


# style CSS appliqué à la page
pageStyleStr = """
<style>
    /* s'applique au titre écrit dans la page HTML */
    h1 {
        text-align: center;
    }
    /* s'applique à l'image contenue dans la div dont l'attribut data-testid vaut "stImage" */
    /*
    div[data-testid="stImage"]>img {
        display: inline-block;
        display: table;
        display: flex;
        justify-content: center;
        max-width: 100%;
        height: fit-content;
        width: fit-content;
        width: intrinsic;
    }
    */
</style>
"""
st.write(pageStyleStr, unsafe_allow_html=True)

image = Image.open("./src/front-end/resources/formIAble_logo.png")
with st.columns([1, 10, 1])[1]:
    # affiche l'image dans l'application
    st.image(np.array(image))

st.title("Application d'analyse automatique de formulaires CERFA")
