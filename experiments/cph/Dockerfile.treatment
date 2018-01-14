FROM floydhub/dl-docker:cpu

RUN \
  echo "h5py==2.7.0\n\
        lifelines==0.9.4\n\
        logger==1.4\n\
        tensorboard-logger==0.0.3\n\
        numpy>=1.9.0\n\
        matplotlib==2.0.0" > /requirements.txt && \
  pip install -r /requirements.txt

COPY . /

CMD [ "python", "-u", "/scripts/cph_run.py", \
"treatment", \
"/shared/data/sim_treatment_dataset.h5", \
"--results_dir", "/shared/results/"]