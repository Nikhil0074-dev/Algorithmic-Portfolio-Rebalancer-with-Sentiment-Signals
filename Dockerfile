FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
