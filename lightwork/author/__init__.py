"""Licensed source authoring and crop-first interactive layouts.

Author: ``lightwork author source.md --slug example --licence original --attribution 'Your name' --privacy-confirmed --backend glimmer``.
Set LW_CHAT_BASE_URL/LW_CHAT_MODEL for real Glimmer. The 180s deadline can be deliberately extended with --timeout.
Calls, raw failures and the unchanged model graph are preserved in authoring/<timestamp>; --human-edit saves a diff.
URL ingestion removes scripts/navigation; PDF ingestion requires Poppler pdftotext. Only original/CC BY material is accepted.

Layout: ``lightwork layout procedures/example/stepgraph.json --manual --work-area X Y W H --mode normal``.
Open /screen, capture a measured crop, choose each object, drag zones or click extreme grid CELL CENTRES TL/TR/BR/BL,
then accept. No image leaves the selected work area. Omit --manual to explicitly request Spark/SAM suggestions.
Manual sam.mode is the future noun prompt, validated=False; successful SAM calls record validated=True.
A SAM plate silhouette cannot locate centre cells: it always requires explicit centre clicks. Outputs default to
.lightwork/layouts; raw layout calls and measured crops live in private .lightwork/layout-sessions.
"""
