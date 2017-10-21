# DeepSurv Experiments

Experiments are run using Docker containers built off of the [floydhub](https://github.com/floydhub/dl-docker) deep learning Docker images. 

### Requirements

Download and install [Docker](https://www.docker.com/community-edition#/download). If you plan to use FloydHub's GPU tag, install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
	
### Experiments

To run one of the experiments from the paper, use the following command from this directory:

	 EXPERIMENT=${EXPERIMENT_ID} docker-compose up --build

The following experiments are provided:

| Experiment    | `${EXPERIMENT_ID} `|
| ------------- |:-------------:|
| Simulated Linear Data | `linear` |
| Simulated Nonlinear (Gaussian) Data | `gaussian` |
| Worchester Heart Attack Study (WHAS) | `whas` |
| Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) | `support` |
| Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) | `metabric` |
| Simulated Treatment Data | `treatment` |
| Rotterdam & German Breast Cancer Study Group (GBSG) | `gbsg` |

The hyper-parameters for each experiment are in this repo's directory: `DeepSurv/experiments/deepsurv/models/`

For more details on each experiment, reference [DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network](http://arxiv.org/abs/1606.00931).

#### Using GPU

If you have nvidia-docker installed, you can run DeepSurv experiments using your GPU. To do so change the tag in the first line of your experiment's docker file. 

For example, to run the simulated linear data experiment with the GPU change the first line the file `./deepsurv/Docker.linear` to:

	FROM floydhub/dl-docker:gpu	
