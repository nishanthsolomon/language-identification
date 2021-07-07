FROM python:3.8.3
EXPOSE 8000

WORKDIR /language-identification

ENV PYTHONPATH=/language-identification

COPY app app
COPY language_identification language_identification
COPY requirements.txt ./

RUN pip install --no-cache-dir -U pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app/main.py"]