FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN apt-get update
RUN apt-get install build-essential wget unzip -y
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
ENV PYTHONPATH /workspace:${PYTHONPATH}
