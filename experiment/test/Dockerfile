FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y openssh-client bash && \
    apt-get clean

WORKDIR /app

RUN mkdir -p /root/.ssh
COPY keep_ssh_tunnel.sh ./
COPY .env ./

RUN chmod +x keep_ssh_tunnel.sh

RUN pip install --upgrade pip
RUN pip install pydantic ollama

COPY test.py .

CMD ["bash", "-c", "./keep_ssh_tunnel.sh & sleep 5 && python3 test.py"]