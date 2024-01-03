FROM python

RUN apt update -y && \
    cd /home && \
    git clone "https://github.com/celleste-bio/public-cocoa.git"  && \
    python -m venv /home/pyenv

COPY bash_configs.sh /bash_configs.sh
RUN cat /bash_configs.sh >> ~/.bashrc

WORKDIR /home

CMD ["/bin/bash"]