# paths
PY=./.venv/bin/python
PIP=./.venv/bin/pip
BACKEND=backend
ARTIFACTS=$(BACKEND)/artifacts
CONFIG=$(BACKEND)/training/config/cicids.yaml

# venv
init:
	cd $(BACKEND); \
		python3 -m venv .venv; \
		$(PY) -m pip install -U pip setuptools wheel; \
		$(PIP) install --only-binary=:all: -r requirements.txt

# training
train-all:
	cd $(BACKEND) && $(PY) -m training.cli train-all --config training/config/cicids.yaml --out artifacts

train-xgb:
	cd $(BACKEND) && $(PY) -m training.cli train-xgb --config training/config/cicids.yaml --out artifacts

train-lgbm:
	cd $(BACKEND) && $(PY) -m training.cli train-lgbm --config training/config/cicids.yaml --out artifacts

train-cat:
	cd $(BACKEND) && $(PY) -m training.cli train-cat --config training/config/cicids.yaml --out artifacts

lccde:
	cd $(BACKEND) && $(PY) -m training.cli build-lccde --config training/config/cicids.yaml --out artifacts

# serving
serve:
	cd $(BACKEND) && IDSML_ARTIFACTS_DIR=$(PWD)/$(ARTIFACTS) $(PY) app.py

# docker
docker-build:
	docker build -t idsml-ide-backend:latest $(BACKEND)

docker-run:
	docker run --rm -p 8000:8000 \
	  -e IDSML_ARTIFACTS_DIR=/app/artifacts \
	  -v "$(PWD)/$(ARTIFACTS)/cicids:/app/artifacts:ro" \
	  idsml-ide-backend:latest

# utility
clean:
	rm -rf $(BACKEND)/.venv $(ARTIFACTS) __pycache__ $(BACKEND)/**/__pycache__

# schema

schema:
	@curl -s http://127.0.0.1:8000/schema | jq
