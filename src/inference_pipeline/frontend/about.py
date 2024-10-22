import streamlit as st
from streamlit_extras.colored_header import colored_header
from src.setup.paths import IMAGES_DIR


colored_header(label=":violet[About Me]", description="", color_name="green-70")


col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(str(IMAGES_DIR/"profile.jpeg"), width=300)

st.markdown(
    """
    Hi there! My name is Kobina, and I'm a mathematician and Machine Learning Engineer.
    
    Thanks for using the service. You can view its code [here](https://github.com/kobinabrandon/Hourly-Divvy-Trip-Predictor).

    Making this application was quite the adventure, and it's been a very rewarding experience that has taught me a great deal. 
    I'll continue to maintain and improve the system, as Divvy continue to provide data on a monthly basis. 

    I'm always contemplating new ideas for machine learning systems to build, and looking for ways to become a better 
    engineer. Having completed the MVP (minimal viable product) for this project, I'm now shifting my focus to: 
    
    - completing an NLP system trained on a corpus of books to produce models that use RAG (retrieval augmented generation) to 
    teach the user about the life and ideas of one of my personal heroes. 

    - completing a second NLP system that provides language translation services, using primarily pretrained transformers.

    - completing a system that provides predictions of exchange rates from various major currencies to (and from) various 
    African currencies.
    
    - incorporating the Rust programming language into my work, so that I can build more performant, and memory-safe 
    MLOps and LLMOps projects. 
    
    Machine learning models deliver the most value when they are integrated into end-to-end applications with high 
    quality data preprocessing, model training, and inference pipelines. If you have data, and some ideas about how a 
    machine learning application can be built around it to bring value to your business, consider reaching out to me.
    
    You can find a link to my Github [here](http://github.com/kobinabrandon), and my 
    LinkedIn [here](https://www.linkedin.com/in/kobina-brandon-aa9445134).
    """
)
