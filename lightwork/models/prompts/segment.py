"""SAM 3.1 request shape (sam__segmenting.md). The prompt is the noun itself or the object-prompt grammar."""

from __future__ import annotations

VERSION = "segment-v1"
MAX_OBJECTS = 16  # server limit for box/point prompts

# Metadata values are strings. one_bit masks are several times smaller than lossless (box/point prompts always
# return one_bit); the decoded raster is the same binary mask.
METADATA = {"include_confidence": "true", "mask_encoding": "one_bit"}

# One concept per request: a list joined by these returns zero matches (sam__segmenting.md "concept prompts").
CONCEPT_SEPARATORS = (",", ";", "+", "\n")
