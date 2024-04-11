#Image: ecolley3/ml-damage-api

FROM python:3.11

RUN pip install Flask==3.0
RUN pip install tensorflow==2.15

COPY proj_models /proj_models
COPY proj_api.py /proj_api.py


CMD ["python", "proj_api.py"]