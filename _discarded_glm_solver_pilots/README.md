Discarded 2026-08-11: GLM 5.2 as RAG-pilot solver produced systematically truncated output
(49.6% of the original 88-query pilot's answers hit SOLVER_MAX_TOKENS=8192 mid-derivation; three
follow-up attempts to fix it — doubling the cap, an explicit concise-solving instruction, and both
together — got no better than 50% truncated, one attempt made it worse). See
`results/rag_pilot.md`'s correction banner and `results/JOURNEY_LOG.md` (2026-08-11 entries) for
the full diagnosis. Solver switched to DeepSeek for the utility-curve rerun. Kept here, not
deleted, in case the raw truncated responses are useful for future truncation-detection work.
