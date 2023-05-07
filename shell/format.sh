#!/bin/bash
isort --sl gradient_accumulator
black --line-length 80 gradient_accumulator
flake8 gradient_accumulator