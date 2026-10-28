# Executed validation

2026-10-28, Python 3.12:

- Windows: 23 tests passed.
- Docker Linux: build passed; 23 tests passed with network disabled.
- Offline demo without approval: booking returned `confirmation_required`.
- Offline demo with consent and `slot-1` approval: synthetic booking succeeded.
- `pip check`: no broken requirements.
- OpenAI adapter: request contract checked with a fake SDK client. No paid API call.

The Docker test environment emitted a Starlette deprecation warning for its httpx
TestClient transport. All tests passed; a future dependency update may require
migrating the test transport. No clinical, production-load, or model-quality
benchmark is claimed.
