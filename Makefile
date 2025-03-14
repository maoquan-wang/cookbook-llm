

## Check and install required formatting tools
check-tools:
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && uv add isort)
	@command -v black >/dev/null 2>&1 || (echo "Installing black..." && uv add black)


## Format modified Python files using isort and black
format: check-tools
	@echo "Formatting modified Python files..."
	git diff --name-only --diff-filter=M | grep '\.py$$' | xargs -I {} sh -c 'isort {} && black {}'
