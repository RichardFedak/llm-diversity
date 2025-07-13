#!/bin/bash

CONTAINER_NAME="web_server"
DB_PATH_IN_CONTAINER="/app/instance/db.sqlite"
LOCAL_DB="prod_db.sqlite"
OUTPUT_DIR="reporting"
INTERACTION_CSV="${OUTPUT_DIR}/interaction.csv"
PARTICIPATION_CSV="${OUTPUT_DIR}/participation.csv"

# Create reporting folder if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Copy the SQLite DB from the container
docker cp "${CONTAINER_NAME}:${DB_PATH_IN_CONTAINER}" "${LOCAL_DB}"

# Export tables to CSV
sqlite3 "${LOCAL_DB}" <<EOF
.headers on
.mode csv

.output ${INTERACTION_CSV}
SELECT * FROM interaction;

.output ${PARTICIPATION_CSV}
SELECT * FROM participation;

.output stdout
.exit
EOF

echo "Export completed:"
echo " - ${INTERACTION_CSV}"
echo " - ${PARTICIPATION_CSV}"
