FROM python:3.12.5-slim-bookworm

EXPOSE 8501

ENV EMAIL=email 
ENV COMET_API_KEY=some_code
ENV COMET_PROJECT_NAME=some_name
ENV COMET_WORKSPACE=some_other_name
ENV HOPSWORKS_API_KEY=some_other_code
ENV HOPSWORKS_PROJECT_NAME=who_knows

RUN apt-get update && apt-get install -y --no-install-recommends pip gcc make python3-dev
RUN pip install poetry

WORKDIR /app

COPY . /app/
COPY data/ /data/
COPY  models/ /models/

VOLUME /data
VOLUME /models

RUN poetry install 

ENTRYPOINT ["/bin/bash", "-c", "poetry", "run", "streamlit", "run", "src/inference_pipeline/frontend/main.py", "--server.port", "8501"]
