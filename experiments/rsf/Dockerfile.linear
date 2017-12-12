FROM floydhub/dl-docker:cpu

RUN \
  apt-get update -qq && \
  apt-get install -y \
                     lsb-release && \
  echo "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse\n" \
      >> /etc/apt/sources.list.d/added_repos.list && \
  echo "deb http://cran.cnr.berkeley.edu/bin/linux/ubuntu $(lsb_release -sc)/" \
      >> /etc/apt/sources.list.d/added_repos.list && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
  apt-get update -qq && \
  apt-get install -y \
                     aptdaemon \
                     ed \
                     git \
         mercurial \
         libcairo-dev \
         libedit-dev \
         python3 \
         python3-pip \
         r-base \
         r-base-dev \
         sudo \
         wget &&\
  rm -rf /var/lib/apt/lists/*

RUN \
  echo "rPython\n\
        randomForestSRC" > rpacks.txt && \
  R -e 'install.packages(sub("(.+)\\\\n","\\1", scan("rpacks.txt", "character")), repos="http://cran.cnr.Berkeley.edu")' && \
  rm rpacks.txt

RUN \
  echo "h5py==2.7.0\n\
        lifelines==0.9.4\n\
        logger==1.4\n\
        tensorboard-logger==0.0.3\n\
        numpy>=1.9.0\n\
        matplotlib==2.0.0\n\
        rpy2==2.8.3" > /requirements.txt && \
  pip install -r /requirements.txt

CMD [ "python", "-u", "/scripts/rsf_run.py", \ 
"/shared/data/linear_survival_data.h5", \
"--results_dir", "/shared/results/", \
"--num_trees", "100"]