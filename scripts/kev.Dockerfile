# Kev decision-model server for Tesla M10 (sm_50): Python 3.12 + torch 2.8 cu126 (kev pins torch<2.9).
FROM python:3.12-slim
RUN pip install --no-cache-dir "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126
COPY kev /opt/kev
RUN pip install --no-cache-dir "/opt/kev[serve]"
WORKDIR /opt/kev
