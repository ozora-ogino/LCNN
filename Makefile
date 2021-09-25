init: # Setup pre-commit
	pip3 install -r ./requirements.txt
	pre-commit install --hook-type pre-commit --hook-type pre-push

lint: # Lint all files in this repository
	pre-commit run --all-files --show-diff-on-failure

test: # Run tests
	pytest tests.py