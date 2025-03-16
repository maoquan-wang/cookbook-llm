

## Check and install required formatting tools
check-tools:
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && uv add isort)
	@command -v black >/dev/null 2>&1 || (echo "Installing black..." && uv add black)

## Check Node env
check-nodejs:
	@echo "Checking Node Enviroment: npm, node "
	@command -v node >/dev/null 2>&1 || (echo "Please install node.js to continue. (ref: https://github.com/nvm-sh/nvm)")

	@if command -v npm >/dev/null 2>&1; then \
		echo "npm version: $(shell npm --version)"; \
		echo "node version: $(shell node --version)"; \
	else \
		exit 1; \
	fi

## Format modified Python files using isort and black
format: check-tools
	@echo "Formatting modified Python files..."
	git diff --name-only --diff-filter=M | grep '\.py$$' | xargs -I {} sh -c 'isort {} && black {}'
