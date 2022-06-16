FROM python:3.7
WORKDIR /app


ENV FLASK_APP=main.py
ENV FLASK_ENV=development

RUN python3 --version
RUN pip3 --version

COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
CMD python main.py