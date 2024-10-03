import streamlit as st
from streamlit_extras.colored_header import colored_header


colored_header(label=":violet[About Me]", description="", color_name="green-70")

st.markdown(
    """
    Hi there! Thanks for using the service. My name is Kobina, and I'm a mathematician turned Machine Learning Engineer.
    
    Making this application was quite the adventure, and  it's been a deeply rewarding experience that I've learned 
    a great deal from. I'll continue to maintain and improve the system, as Divvy continue to provide data on a monthly
    basis.

    I'm always dreaming up new end-to-end ML systems to build, and looking for ways to become a better engineer. 
    Having completed the MVP (minimal viable product) for this project, I'm now shifting my focus to: 
    
    - completing an NLP system that uses a RAG (retrieval augmented generation) pipeline to produce models that teach 
    the user about one of my personal heroes. 

    - completing a second NLP system that provides language translation services, using primarily pretrained LLMs.
    
    - incorporating the Rust programming language into my work, so that I can build more performant, and memory-safe 
    MLOps and LLMOps projects. 
    
    Machine learning models deliver the most value when they are integrated into end-to-end applications with high 
    quality preprocessing, training, and inference pipelines, and I take great pride in my ongoing dedication to 
    this excellent craft.
    """
)


st.markdown("""You can find links to my Github & LinkedIn profiles below""")

links = """
<a href="http://github.com/kobinabrandon" target='_blank'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/c/c6/Font_Awesome_5_brands_github-square.svg' style='width:50px; height:50px;'>
</a> 

<a href="https://www.linkedin.com/in/kobina-brandon-aa9445134" target='_blank'>
    <img src="https://pngmind.com/wp-content/uploads/2019/08/Linkedin-Logo-Png-Transparent-Background-1.png" style='width:50px; height:50px;'>
</a>
"""

st.markdown(body=links, unsafe_allow_html=True)
