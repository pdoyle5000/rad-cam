#!/bin/bash
isort -rc .
black *.py radcam/*.py
pytest --cov-report term-missing --cov=.
mypy radcam/*.py
