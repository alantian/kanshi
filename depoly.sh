#!/bin/bash

# Build the docker image.
docker build -t local:kanshi .

# Start the container.
docker run -d --restart always -p 23334:8002 local:kanshi     # for production

