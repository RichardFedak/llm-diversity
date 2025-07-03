FROM debian:bullseye-slim

RUN apt-get update && \
    apt-get install -y openssh-client bash && \
    apt-get clean

WORKDIR /app

RUN mkdir -p /root/.ssh
COPY known_hosts /root/.ssh/known_hosts
COPY keep_ssh_tunnel.sh ./
COPY .env ./

RUN chmod +x keep_ssh_tunnel.sh

CMD ["./keep_ssh_tunnel.sh"]
