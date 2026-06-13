# BCI Platform — Makefile
# ========================
# Usage:
#   make install       — install Python dependencies
#   make demo          — run standalone simulation demo
#   make benchmark     — run DSP latency benchmark
#   make validate      — run ISO 14155 validation
#   make train         — train all ML models
#   make api           — start FastAPI server
#   make dashboard     — start Plotly Dash dashboard
#   make pipeline      — run full BCI pipeline (simulation)
#   make test          — run all tests with coverage
#   make docker-up     — start Docker infrastructure
#   make docker-down   — stop Docker infrastructure
#   make kafka-setup   — create Kafka topics
#   make lint          — run flake8 linter
#   make format        — run black formatter
#   make clean         — remove build artifacts

.PHONY: all install demo benchmark validate train api dashboard pipeline test docker-up docker-down kafka-setup lint format clean

all: install

# ── Setup ─────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Demo & validation (no hardware required) ──────────────────────
demo:
	python scripts/demo_simulation.py

benchmark:
	python scripts/benchmark_latency.py

validate:
	python scripts/validate_iso14155.py

# ── ML training ───────────────────────────────────────────────────
train:
	python ml/train.py --model compare
	python ml/train.py --model cnn --epochs 50
	python ml/train.py --model eegnet --epochs 50

train-cnn:
	python ml/train.py --model cnn --epochs 50

train-compare:
	python ml/train.py --model compare

# ── Services ──────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	python dashboard/app.py

pipeline:
	python scripts/run_pipeline.py --simulate --paradigm motor_imagery

pipeline-p300:
	python scripts/run_pipeline.py --simulate --paradigm p300

# ── Testing ───────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --timeout=30

test-dsp:
	pytest tests/test_dsp.py -v

test-ml:
	pytest tests/test_ml.py -v

test-security:
	pytest tests/test_security.py -v

test-resilience:
	pytest tests/test_resilience.py -v

test-api:
	pytest tests/test_api.py -v

# ── Docker ────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d zookeeper kafka postgres
	@echo "Waiting 15s for Kafka..."
	sleep 15
	$(MAKE) kafka-setup

docker-down:
	docker-compose down

kafka-setup:
	bash scripts/setup_kafka.sh

# ── Code quality ──────────────────────────────────────────────────
lint:
	flake8 . --max-line-length=100 --exclude=venv,__pycache__,.git

format:
	black . --line-length=100 --exclude=venv

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "Clean complete"