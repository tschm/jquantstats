## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DEFAULT_AI_MODEL=claude-sonnet-4.6
LOGO_FILE=.rhiza/assets/rhiza-logo.svg
GH_AW_ENGINE ?= copilot  # Default AI engine for gh-aw workflows (copilot, claude, or codex)
MKDOCS_EXTRA_PACKAGES = --with "mkdocstrings[python]"

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk

## Custom targets

##@ Quality

.PHONY: test-kaleido
test-kaleido: install ## run kaleido static image export tests (requires jquantstats[plot])
	@printf "${BLUE}[INFO] Running kaleido static image export tests...${RESET}\n"
	@${UV_BIN} run pytest -m kaleido -v tests/

.PHONY: changelog
changelog: ## generate/update CHANGELOG.md from git history using git-cliff
	@printf "${BLUE}[INFO] Generating CHANGELOG.md with git-cliff...${RESET}\n"
	@${UVX_BIN} git-cliff --config .github/cliff.toml --output CHANGELOG.md
	@printf "${GREEN}[OK] CHANGELOG.md updated.${RESET}\n"

# Ignore CVE-2026-4539 (ReDoS in pygments AdlLexer) until pygments>=3.3 is on PyPI.
PIP_AUDIT_ARGS = --ignore-vuln CVE-2026-4539

post-license:: ## generate LICENSES.md dependency report
	@printf "${BLUE}[INFO] Generating LICENSES.md dependency report...${RESET}\n"
	@${UV_BIN} run --with pip-licenses pip-licenses --format=markdown --output-file=LICENSES.md
	@printf "${GREEN}[OK] LICENSES.md generated.${RESET}\n"


