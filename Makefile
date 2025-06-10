#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = rl_ids
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"

#################################################################################
# PROJECT SCRIPTS AND COMMANDS                                                  #
#################################################################################

## Preprocess downloaded data
.PHONY: preprocess-data
preprocess-data:
	$(PYTHON_INTERPRETER) -m rl_ids.dataset --raw-dir data/raw/MachineLearningCVE
	$(PYTHON_INTERPRETER) -m rl_ids.normalize
	$(PYTHON_INTERPRETER) -m rl_ids.balance

## Train the DQN model
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m rl_ids.modeling.train

## Evaluate the trained model
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) -m rl_ids.modeling.evaluate

## Start the FastAPI server
.PHONY: api
api:
	$(PYTHON_INTERPRETER) run_api.py

## Start API server in development mode
.PHONY: api-dev
api-dev:
	$(PYTHON_INTERPRETER) run_api.py --reload --log-level debug

## Test the API endpoints
.PHONY: api-test
api-test:
	$(PYTHON_INTERPRETER) -m api.client

## Check if model exists
.PHONY: check-model
check-model:
	@if [ ! -f "models/dqn_model_final.pt" ]; then \
		echo "❌ Model file not found. Please train the model first with 'make train'"; \
		exit 1; \
	else \
		echo "✅ Model file found"; \
	fi

## Full pipeline: install, train, evaluate, and test API
.PHONY: pipeline
pipeline: requirements preprocess-data train evaluate api-test

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

