#!/bin/bash

docker stop public-cocoa
docker rm public-cocoa
docker image remove public-cocoa

docker build --tag public-cocoa .
docker run \
    --detach --tty \
    --name public-cocoa \
    --hostname public-cocoa \
    --publish 8080:8080 \
    public-cocoa \
    bash