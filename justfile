lint: fmt
	ruff check src/imageomics

fmt:
	ruff format src/imageomics

test: lint
	pytest src/imageomics/
