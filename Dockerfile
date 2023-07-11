FROM python:3.10.6-buster

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY gcnb_pkg gcnb_pkg
COPY setup.py setup.py
COPY pickle pickle
RUN pip install .

CMD uvicorn gcnb_pkg.api.fast_get:app --host 0.0.0.0 --port $PORT
