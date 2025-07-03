FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y openssh-client bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /root/.ssh
COPY keep_ssh_tunnel.sh ./
COPY .env ./

RUN chmod +x keep_ssh_tunnel.sh

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 5000
