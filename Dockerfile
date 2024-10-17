FROM python:3.12.5-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends pip gcc make python3-dev cron supervisor
RUN pip install poetry

WORKDIR /app
COPY . /app/
RUN rm -rf /app/supervisord.conf
COPY supervisord.conf /etc/supervisord.conf

RUN poetry install 

EXPOSE 8501

ENV EMAIL=email 
ENV COMET_API_KEY=some_code
ENV COMET_PROJECT_NAME=some_name
ENV COMET_WORKSPACE=some_other_name
ENV HOPSWORKS_API_KEY=some_other_code
ENV HOPSWORKS_PROJECT_NAME=who_knows

# Prevents the streamlit app from asking for an email address
RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

RUN echo "0 * * * * export PATH=$PATH:/usr/local/bin && cd /app/ && make backfill-predictions >> /var/log/cron.log 2>&1" >> /etc/cron.d/backfill_cron_job
RUN chmod 0644 /etc/cron.d/backfill_cron_job
RUN crontab /etc/cron.d/backfill_cron_job

RUN touch /var/log/cron.log /var/log/supervisord.log
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
