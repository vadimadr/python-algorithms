.PHONY: clean clean-test clean-pyc clean-build docs

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

BROWSER := bash -c 'open $$0 || xdg-open $$0 || sensible-browser $$0 || x-www-browser $$0'

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -f coverage.xml
	rm -f junit.xml

lint: ## check style with flake8
	flake8 algorithms tests 

format:
	autoflake -ir algorithms --exclude="algorithms/_extensions,submodules"
	isort . 
	black . --extend-exclude="submodules|.*_extensions.*"

test: ## run tests quickly with the default Python
	pytest


coverage-report: ## check code coverage quickly with the default Python
	pytest --cov=algorithms --junitxml=junit.xml
	coverage html
	coverage xml

coverage: coverage-report
	$(BROWSER) htmlcov/index.html


publish: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

develop: clean ## Conifure python environment for develooping this package
	pip install -r requirements-dev.txt
	python setup.py develop

PLATNAME := $$(python -c "import distutils.util as d; import sys; print(d.get_platform() + '-%d.%d' % sys.version_info[:2])")

ext:
	cmake --build build/temp.$(PLATNAME)
