FROM ghcr.io/broadinstitute/ml4h:tf2.9-latest-cpu

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ecg2af-streamlit-app.py"]
