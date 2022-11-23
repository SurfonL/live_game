import streamlit as st
import cv2
from nav import nav_page


st.set_page_config(
    page_title="Home page",
    page_icon="👋",
    initial_sidebar_state="collapsed"
)
st.write("# Welcome")

# st.sidebar.success("Select a above.")
image = cv2.imread('jinx.png')
skin1, skin2, skin3, skin4 = cv2.imread('skin1.png'), cv2.imread('skin2.png'), cv2.imread('skin3.png'), cv2.imread('skin4.png')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["arcane", "classic", "pink-bomb", "green-bomb", "chaos"])

with tab1:
   st.header("arcane")
   st.image(image)

with tab2:
   st.header("classic")
   st.image(skin1)

with tab3:
   st.header("pink-bomb")
   st.image(skin2)

with tab4:
   st.header("green-bomb")
   st.image(skin3)

with tab5:
   st.header("chaos")
   st.image(skin4)


if st.button("show monsters around me"):
    st.button("Show monsters around me", on_click= nav_page("maps"))
