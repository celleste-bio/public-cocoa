#!/bin/bash

docker stop public-cocoa
docker remove public-cocoa
docker image remove public-cocoa

docker build --tag public-cocoa .
docker run \
    --detach --tty \
    --name public-cocoa \
    --hostname public-cocoa \
    public-cocoa \
    bash