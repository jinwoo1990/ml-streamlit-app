FROM python:3.7-stretch
WORKDIR /api/
RUN mkdir -p /api/logs
COPY ./requirements.txt /api/
COPY ./app.py /api/
COPY ./helper.py /api/
COPY ./pipeline.py /api/
COPY ./api_init.py /api/
COPY ./model_objects.pkl /api/
COPY ./base_logger.py /api/
COPY ./logger_config.json /api/
ENV VIRTUAL_ENV=/api/.venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "app.py"]
