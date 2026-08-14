setup:
	uv sync --locked

notebook:
	uv run jupyter lab

test:
	uv run pytest
