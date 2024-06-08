#!/bin/bash

# Check if .venv directory exists.
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi

# Check if docker container exists.
if [ ! "$(docker ps -q -f name=pgvector)" ]; then
  if [ "$(docker ps -aq -f status=exited -f name=pgvector)" ]; then
    docker rm pgvector
  fi
  docker run -d \
    -e POSTGRES_DB=ai \
    -e POSTGRES_USER=ai \
    -e POSTGRES_PASSWORD=ai \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v pgvolume:/var/lib/postgresql/data \
    -p 5532:5432 \
    --name pgvector \
    phidata/pgvector:16
fi

# Activate virtual environment.
source .venv/bin/activate

# Run the application.
python core/assistant.py