FROM python:3.10-slim

WORKDIR /app/nn_project

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/VerVelVel/nn_project.git .

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch torchvision streamlit
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Hello.py", "--server.port=8501", "--server.address=0.0.0.0"]