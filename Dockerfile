FROM python

RUN apt update -y && \
    cd home && \
    git clone "https://github.com/celleste-bio/public-cocoa.git" 
    #python envirment

WORKDIR /home/public-coca 

CMD ["/bin/bash"]


