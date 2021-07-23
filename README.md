# Applying Clean Code Principles

## Overview
In this project, I applied the clean code principles (modular, documented, and tested) to implement a classifier that predicts customer churn. 

## Dependencies
- [Anaconda](https://www.anaconda.com)
	- Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to the next step.
	- Create local environment
		```
		conda create -n env python=3.9
		conda activate env
		```

		At this point your command line should look something like: `(env) <User>:USER_DIR <user>$`. The `(env)` indicates that your environment has been activated, and you can proceed with further package installations.

## Project Steps
- To automatically generate the `requirements.txt` file, type `pipreqs --force` in cli.
- To install the required python packages, type `pip install -r requirements.txt` in cli.
- To lint a python script, type `black --line-length 79 PATH_TO_FILE --experimental-string-processing` and `flake8 PATH_TO_FILE` in cli.
- To run the machine learning pipeline, type `python churn_library.py` in cli.
- To run the tests, type `python test_churn_library.py` in cli.
- To view the logs, type `cat logs/churn_library.log` in cli. 

