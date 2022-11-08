FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["q6_predict.py", "./"]

EXPOSE 9898

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9898", "q6_predict:app"]