run:
	uv run src

clean:
	find . -not -path "./.venv/*" -name "__pycache__" -type d -exec rm -rf {} \;
