import streamlit as st
from streamlit_extras.colored_header import colored_header


colored_header(label="About Me", description="", color_name="green-70")

st.markdown(
    """
    Hi there! Thanks for using the service. My name is Kobina Brandon, and I'm a Machine Learning Engineer from Ghana/Britain.
    Making this application was quite the adventure, and it took a lot more time and effort that I initially expected. All the 
    same, it's been a deeply rewarding experience that I've learned a great deal from.

    I'm always dreaming up new end-to-end ML systems to build, and looking for ways to become a better engineer. 
    I'm open to doing contract work. 
    
    """
)


st.link_button(
    label="View my Github profile here to look more closely at my work.", 
    url="http://github.com/kobinabrandon"
)
