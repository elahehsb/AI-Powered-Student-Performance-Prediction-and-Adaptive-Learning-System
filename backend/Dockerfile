FROM python:3.8-slim

RUN apt-get update && apt-get install -y python3-dev gcc libpq-dev

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
