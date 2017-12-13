FROM floydhub/dl-docker:gpu

RUN pip install https://github.com/Lasagne/Lasagne/archive/master.zip
RUN \
  echo "h5py==2.7.0\n\
        lifelines==0.9.4\n\
        logger==1.4\n\
        Optunity==1.1.1\n\
        tensorboard-logger==0.0.3\n\
        matplotlib==2.0.0" > /requirements.txt && \
  pip install -r /requirements.txt

COPY . /

ENV THEANO_FLAGS=device=gpu7

CMD [ "python", "-u", "/hyperparam_search.py", \
"/shared/logs", \
"gaussian", \
"/box_constraints.0.json", \
"50", \
"--update_fn", "adam", \
"--num_epochs", "500", \
"--num_fold", "3" ]
