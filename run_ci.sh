#!/bin/bash
pytest --cov-report term-missing --cov=.
mypy radcam/*.py
