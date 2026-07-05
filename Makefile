setup:
	uv sync

notebook:
	uv run --with jupyter jupyter lab

test:
	uv run pytest
