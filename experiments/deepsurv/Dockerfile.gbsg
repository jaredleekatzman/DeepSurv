FROM floydhub/dl-docker:cpu

RUN pip install https://github.com/Lasagne/Lasagne/archive/master.zip

RUN \
  echo "h5py==2.7.0\n\
        lifelines==0.9.4\n\
        logger==1.4\n\
        tensorboard-logger==0.0.3\n\
        numpy>=1.9.0\n\
        matplotlib==2.0.0" > /requirements.txt && \
  pip install -r /requirements.txt

COPY . /

CMD [ "python", "-u", "/scripts/deepsurv_run.py", \
"gbsg", \
"/models/gbsg_model_selu_revision.0.json", \
"/shared/data/gbsg_cancer_train_test.h5", \
"--update_fn","adam",\
"--treatment_idx", "0", \
"--results_dir", "/shared/results/", \
"--num_epochs", "1000"]
