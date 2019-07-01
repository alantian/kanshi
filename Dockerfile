FROM ubuntu:16.04

#-------------------------------------------------------------------------------
### Enable UTF8 in docker instance
#-------------------------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y locales && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
#-------------------------------------------------------------------------------

ADD . /kanshi

WORKDIR /kanshi

RUN apt-get update -y && \
    apt-get install -y coreutils python3 python3-pip wget && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install -r pip_freeze

RUN wget https://storage.googleapis.com/store.alantian.net/depot/scratch_pack.tar && \
    tar xvf scratch_pack.tar

RUN rm -rf dropbox scratch scratch_pack.tar

RUN mv scratch_pack scratch

WORKDIR /kanshi/code/crnnlm/

EXPOSE 8002
CMD [ "bash", "./run_webapi.sh" ]
