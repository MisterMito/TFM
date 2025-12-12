#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TFM
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Sync Python dependencies (uv.lock)
.PHONY: requirements
requirements:
	uv sync

## Sync deps exactly as locked (CI-like)
.PHONY: requirements_frozen
requirements_frozen:
	uv sync --frozen

## Pull LFS files (data/models tracked with LFS)
.PHONY: lfs_pull
lfs_pull:
	git lfs pull

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix .
	uv run ruff format .

## Run tests
.PHONY: test
test:
	uv run pytest -q

## Run pre-commit hooks on all files (CI-like)
.PHONY: precommit
precommit:
	uv run pre-commit run --all-files

## Convenience target: run the usual local gating
.PHONY: check
check: lint test

## Set up Python interpreter environment (optional; uv sync usually suffices)
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


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
