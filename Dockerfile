FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /workspace/
RUN apt-get update
RUN curl -sL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g pnpm
RUN git clone --branch custom-frontend https://github.com/ChungYujoyce/chainlit.git \
    && cd chainlit/backend \
    && pip install -e . \
    && pnpm install \
    && pnpm run buildUi


COPY ./requirements.txt .
RUN pip install --upgrade pip \
  && pip uninstall -y transformer-engine \
  && pip uninstall -y apex \
  && pip install -r requirements.txt

