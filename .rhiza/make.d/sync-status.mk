## sync-status.mk - tools for checking Rhiza sync status
# This file is included by the main Makefile

# Declare phony targets
.PHONY: sync-status sync-status-local sync-status-github

##@ Rhiza Sync Status

sync-status-local: ## show date of last Rhiza sync from local git history
	@printf "${BLUE}[INFO] Last Rhiza Sync (Local Git History):${RESET}\n"
	@LAST_SYNC=$$(git log --grep="Sync with rhiza\|Update via rhiza" --format="%ad - %s" --date=format:'%Y-%m-%d %H:%M:%S' -1 2>/dev/null); \
	if [ -n "$$LAST_SYNC" ]; then \
		printf "  ${GREEN}%s${RESET}\n" "$$LAST_SYNC"; \
	else \
		printf "  ${YELLOW}No Rhiza sync commits found in local history${RESET}\n"; \
	fi
	@printf "\n"

sync-status-github: gh-install ## show date of last Rhiza sync from GitHub Actions
	@printf "${BLUE}[INFO] Last Rhiza Sync (GitHub Actions):${RESET}\n"
	@if command -v gh >/dev/null 2>&1; then \
		GH_TEMPLATE='{{range .}}{{if or (eq .conclusion "success") (eq .status "completed")}}  {{printf "✓" | color "green"}} Last successful sync: {{timeago .createdAt | color "green"}} ({{.createdAt}}){{"\n"}}{{else if eq .status "in_progress"}}  {{printf "⏳" | color "yellow"}} Sync in progress (started {{timeago .createdAt}}){{"\n"}}{{else}}  {{printf "✗" | color "red"}} Last run: {{timeago .createdAt}} ({{.conclusion}}){{"\n"}}{{end}}{{end}}'; \
		LAST_RUN=$$(gh run list --workflow=rhiza_sync.yml --limit 1 --json conclusion,createdAt,status --template "$$GH_TEMPLATE" 2>/dev/null); \
		if [ -n "$$LAST_RUN" ]; then \
			printf "%s" "$$LAST_RUN"; \
		else \
			printf "  ${YELLOW}No Rhiza sync workflow runs found${RESET}\n"; \
		fi; \
	else \
		printf "  ${YELLOW}gh CLI not installed. Run 'make gh-install' for instructions.${RESET}\n"; \
	fi
	@printf "\n"

sync-status: sync-status-local sync-status-github ## show date of last Rhiza sync (both local and GitHub)
	@printf "${BLUE}[INFO] Rhiza Version:${RESET}\n"
	@printf "  Current version: ${GREEN}$(RHIZA_VERSION)${RESET}\n"
	@printf "\n"
	@printf "${BLUE}[INFO] To sync now, run:${RESET}\n"
	@printf "  ${BOLD}make sync${RESET}\n"
	@printf "\n"
