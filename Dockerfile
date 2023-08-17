FROM postgres:15

COPY . /tmp/build
RUN (cd /tmp/build && ./docker.sh)
