FROM floydhub/dl-docker:cpu

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

CMD [ "python", "-u", "/hyperparam_search.py", \
"/shared/logs", \
"gaussian", \
"/box_constraints.0.json", \
"100", \
"--update_fn", "adam", \
"--num_epochs", "500", \
"--num_fold", "3" ]
