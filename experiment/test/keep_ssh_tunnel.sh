#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
PID_FILE="$SCRIPT_DIR/ssh_tunnel.pid"

# Load SSH_CONNECTION from .env
if [[ -f "$ENV_FILE" ]]; then
    SSH_CONNECTION=$(grep '^SSH_CONNECTION=' "$ENV_FILE" | cut -d '=' -f2- | tr -d '"')
    export SSH_CONNECTION
fi


if [[ -z "$SSH_CONNECTION" ]]; then
    echo "SSH_CONNECTION not found or empty in .env file."
    exit 1
fi

echo "Keeping SSH tunnel alive: $SSH_CONNECTION"

echo $$ > "$PID_FILE"

# Infinite loop to keep SSH connection alive
while true; do
    echo "Starting SSH tunnel..."
    eval "$SSH_CONNECTION"
    echo "SSH tunnel exited. Reconnecting in 5 seconds..."
    sleep 5
done
