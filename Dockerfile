FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /workspace/
RUN apt-get update && \
 apt-get install -y poppler-utils

COPY ./requirements.txt .
RUN pip install --upgrade pip \
  && pip install -r requirements.txt \
  && pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable \

RUN pnpm install
RUN git clone --branch custom-frontend https://github.com/ChungYujoyce/chainlit.git \
    && cd chainlit/backend \
    && pip install -e . \
    && pnpm run buildUi

