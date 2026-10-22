.PHONY: check test report data all

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy

test:
	uv run pytest

report:
	uv run updown report

data:
	uv run updown data

all: check test report
