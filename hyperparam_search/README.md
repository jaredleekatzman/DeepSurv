# A. Configure hyper-parameter search

To run a hyper-parameter search you need to configure two files:

1. Box constraints
2. Dockerfile

The box constraints is a json file containing a dictionary mapping the hyper-parameter name to the range you want to search over. A default dictionary, with default ranges, is provided as an example.

The dockerfile needs to be configured with the following details:

1. The location of the dataset file
2. The box constraints file to use

Any other parameters to the script are optional and are documented in hyperparam_search.py.
Additionally, the hyper-parameter can be run using the gpu by changing the first import statement to `FROM floydhub/dl-docker:gpu`

# B. Running a hyper-parameter search

1. Configure your hyper-parameter search (see A)
2. From the directory run the command:
	
		docker-compose up --build

3. The resulting optunity log should be saved as a .pkl file with the name: optunity\_log\_UUID.pkl .
