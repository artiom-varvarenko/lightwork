"""Lightwork model layer (C5/C9, D2-D4/D13/D15/D21).

Pure replay-facing API: requests.build_request/PROMPT_VERSIONS; parse.parse_response; pricing.cost_usd and
aggregate_usage. These perform no I/O. ``synthetic`` creates explicitly synthetic response fixtures, never live
answers. ``record.write_call`` writes C5 consistently for recordings and fixtures.

Live API: client.ModelClient.execute is synchronous and intended for a worker thread. Each HTTP attempt uses an
async deadline internally, so an SSE peer cannot extend the total deadline by dribbling bytes. Invalid schema or
length retries are separate derived intents (the reducer's responsibility), never hidden transport repairs.
Usage aggregates across transport retries, whose raw bodies are kept in call-dir attempts/; final body stays at
response.sse/json. ModelAccessError carries the already-recorded model_result for logging before stopping.

Budget.from_settings uses the persisted, cross-process locked .lightwork/spend.json. Context-sized reservations
are intentionally conservative; observed usage releases unused funds. Missing billing evidence remains held,
not invented as spend. Reconcile stranded reservations against call records after a crash.

DevCache.from_settings is opt-in and ModelClient refuses it unless recording=False. Non-Meta models also require
LW_DEV_ALLOW_NON_META=1 and dev-standin provenance; official runs must be written under runs-dev by the recorder.
No SAM stand-ins. API keys stay in transport headers, never in request.json.

Live smoke: ``uv run python -m lightwork.models.smoke --help``. Real fixture calls preserve model provenance even
when their image is synthetic; generated responses use the contracts' synthetic provenance instead.
"""
