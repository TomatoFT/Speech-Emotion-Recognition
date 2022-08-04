FROM python:3

WORKDIR /mnt/d/DS102-Project

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]