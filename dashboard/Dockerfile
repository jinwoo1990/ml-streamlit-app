FROM python:3.7-stretch
WORKDIR /dashboard/
COPY ./requirements.txt /dashboard/
COPY ./webapp.py /dashboard/
ENV VIRTUAL_ENV=/dashboard/.venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "webapp.py"]