"""Versioned prompts and result schemas, one module per role (C9). Same texts and schemas on every backend (D13).

Any change to a role's text or schema must bump that role's ``VERSION`` (``requests.PROMPT_VERSIONS``); a change to
``REPAIR`` bumps every role. Recorded requests are fingerprinted, so an edit shows up as fingerprint diffs in
``verify`` of every run recorded before it (PLAN.md §4.2).
"""

# Follow-up user turn of the one retry after an unusable reply (``ctx["retry"]["reason"] == "invalid"``).
REPAIR = "Your reply could not be used: {error}. Reply again with only the corrected answer in the required format."
