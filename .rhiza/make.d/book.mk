## book.mk - Book-building targets (Zensical / MkDocs fallback)

.PHONY: book zensical-build zensical-serve \
        mkdocs-build mkdocs-serve mkdocs \
        test benchmark stress hypothesis-test \
        _book-reports _book-notebooks

# No-op stubs — overridden by test.mk / bench.mk when present
test:: ; @:
benchmark:: ; @:
stress:: ; @:
hypothesis-test:: ; @:

BOOK_OUTPUT ?= _book

# Detect build configs (zensical takes priority)
_ZENSICAL_CFG := $(wildcard zensical.toml)

# uv run --with zensical keeps the project venv active so jquantstats
# stays importable by mkdocstrings, without requiring zensical in pyproject.toml.
_ZENSICAL := ${UV_BIN} run --with zensical zensical

##@ Book

_book-reports: test benchmark stress hypothesis-test
	@mkdir -p docs/reports
	@for src_dir in \
	  "_tests/html-coverage:reports/coverage" \
	  "_tests/html-report:reports/test-report" \
	  "_tests/benchmarks:reports/benchmarks" \
	  "_tests/stress:reports/stress" \
	  "_tests/hypothesis:reports/hypothesis"; do \
	  src=$${src_dir%%:*}; dest=docs/$${src_dir#*:}; \
	  if [ -d "$$src" ] && [ -n "$$(ls -A "$$src" 2>/dev/null)" ]; then \
	    printf "${BLUE}[INFO] Copying $$src -> $$dest${RESET}\n"; \
	    mkdir -p "$$dest"; cp -r "$$src/." "$$dest/"; \
	  else \
	    printf "${YELLOW}[WARN] $$src not found, skipping${RESET}\n"; \
	  fi; \
	done
	@printf -- "---\nicon: lucide/file-bar-chart-2\n---\n\n# Reports\n\n" > docs/reports.md
	@[ -f "docs/reports/test-report/report.html" ] && echo "- [Test Report](reports/test-report/report.html)"       >> docs/reports.md || true
	@[ -f "docs/reports/hypothesis/report.html" ]  && echo "- [Hypothesis Report](reports/hypothesis/report.html)" >> docs/reports.md || true
	@[ -f "docs/reports/benchmarks/report.html" ]  && echo "- [Benchmarks](reports/benchmarks/report.html)"        >> docs/reports.md || true
	@[ -f "docs/reports/stress/report.html" ]      && echo "- [Stress Report](reports/stress/report.html)"          >> docs/reports.md || true
	@[ -f "docs/reports/coverage/index.html" ]     && echo "- [Coverage Report](reports/coverage/index.html)"      >> docs/reports.md || true

_book-notebooks:
	@if [ -d "$(MARIMO_FOLDER)" ]; then \
	  mkdir -p docs/notebooks; \
	  for nb in $(MARIMO_FOLDER)/*.py; do \
	    [ -f "$$nb" ] || continue; \
	    name=$$(basename "$$nb" .py); \
	    printf "${BLUE}[INFO] Exporting $$nb${RESET}\n"; \
	    abs_output="$$(pwd)/docs/notebooks/$$name.html"; \
	    (cd "$$(dirname "$$nb")" && ${UV_BIN} run marimo export html --sandbox "$$(basename "$$nb")" -o "$$abs_output"); \
	  done; \
	  printf -- "---\nicon: lucide/book-open\n---\n\n# Marimo Notebooks\n\n" > docs/notebooks.md; \
	  for html in docs/notebooks/*.html; do \
	    [ -f "$$html" ] || continue; \
	    name=$$(basename "$$html" .html); \
	    echo "- [$$name](notebooks/$$name.html)" >> docs/notebooks.md; \
	  done; \
	else \
	  printf "${YELLOW}[WARN] MARIMO_FOLDER '$(MARIMO_FOLDER)' not found, skipping notebooks${RESET}\n"; \
	  printf -- "---\nicon: lucide/book-open\n---\n\n# Marimo Notebooks\n\nNo notebooks found.\n" > docs/notebooks.md; \
	fi

book: _book-reports _book-notebooks ## build documentation with Zensical
	printf "${BLUE}[INFO] Building with Zensical...${RESET}\n"; \
	$(_ZENSICAL) build; \
    touch "$(BOOK_OUTPUT)/.nojekyll"; \
	printf "${GREEN}[SUCCESS] Book built at $(BOOK_OUTPUT)/${RESET}\n"; \
