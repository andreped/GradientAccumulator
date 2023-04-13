PYTHON := python
PIP := $(PYTHON) -m pip
SETUP := $(PYTHON) setup.py -q
COVERAGE := coverage
DOCS := docs
SPHINXOPTS := '-W'
RM := rm -rf

.PHONY: clean dev distribute docs help install py_info test

help:
	@ echo "Usage:\n"
	@ echo "make install   Install the package using pip."
	@ echo "make dev       Install the package for development using pip."
	@ echo "make docs      Generate package documentation using Sphinx"
	@ echo "make clean     Remove auxiliary files."

install: clean py_info
	$(PIP) install --upgrade .

dev: clean py_info
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade setuptools wheel

docs: clean
	make -C $(DOCS) html SPHINXOPTS=$(SPHINXOPTS)

clean:
	@ $(RM) $(DOCS)/build $(DOCS)/generated
	@ $(RM) build dist

py_info:
	@ echo "Using $$($(PYTHON) --version) at $$(which $(PYTHON))"
