#!/bin/bash
isort --sl gradient-accumulator
black --line-length 80 gradient-accumulator
flake8 gradient-accumulator