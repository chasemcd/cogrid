# External Integrations

**Analysis Date:** 2026-02-10

## APIs & External Services

**Not applicable** - CoGrid is a reinforcement learning environment library with no external API integrations. The library provides environment interfaces but does not consume external APIs.

## Data Storage

**Databases:**
- Not used - CoGrid is a stateless RL environment library without persistent storage

**File Storage:**
- Local filesystem only - Visualization assets and documentation stored locally
- No cloud storage integration

**Caching:**
- Not used - No caching layer required

## Authentication & Identity

**Not applicable** - CoGrid requires no authentication. It is an offline library for multi-agent RL research.

## Monitoring & Observability

**Error Tracking:**
- None - Standard Python exception handling only

**Logs:**
- None - No logging framework integrated
- Standard output/print statements for interactive mode in `cogrid/run_interactive.py`

## CI/CD & Deployment

**Hosting:**
- PyPI distribution hosted at https://pypi.org/project/cogrid/
- GitHub repository at https://github.com/chasemcd/cogrid

**CI Pipeline:**
- Read the Docs integration (`.readthedocs.yaml`) for automatic documentation builds on commits
- No GitHub Actions or other CI/CD platform detected

**Documentation Hosting:**
- Read the Docs (readthedocs.io) - Configured in `.readthedocs.yaml`
- Build environment: Ubuntu 22.04
- Python version: 3.10
- Auto-builds from `docs/conf.py` using Sphinx

## Environment Configuration

**Required env vars:**
- None - No environment variables required for operation

**Secrets location:**
- Not applicable - No secrets managed

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Inter-Process Communication

**Within Library:**
- PettingZoo `ParallelEnv` interface for agent-environment interaction
- Internal module imports only, no external communication

## Optional Dependencies

**Visualization:**
- PyGame (optional) - Used for rendering grid environments in `render_mode="human"` or `render_mode="rgb_array"` modes
- Imported with try/except in `cogrid/cogrid_env.py:9-11` to allow operation without it

---

*Integration audit: 2026-02-10*
