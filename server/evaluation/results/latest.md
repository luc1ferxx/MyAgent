# Synthetic RAG Evaluation

- Run ID: `2026-04-20T20-44-01-440Z`
- Corpus file: `C:\Users\Jxx\Desktop\agentai\server\evaluation\synthetic-corpus-near-duplicate.json`
- Embedding model: `text-embedding-3-small`
- Chat model: `gpt-5`
- Chunk strategy: `structured`
- Retrieval top-k: `6`
- Compare top-k per doc: `3`
- Chunk size / overlap: `900/180`
- Min relevance score: `0.32`

## Metrics

| Metric | Value |
| --- | ---: |
| Overall pass rate | 1 |
| QA page hit rate | 1 |
| Compare doc coverage | 1 |
| Compare page hit rate | 1 |
| Abstain accuracy | 1 |
| Answer content hit rate | 1 |
| Upload resume success rate | 1 |
| Avg response time (ms) | 14848.88 |
| Avg citation count | 1.63 |
| Resume saved bytes | 2700 |

## Upload Resume Checks

| Document | Chunks | Skipped On Resume | Saved Bytes | Merge OK |
| --- | ---: | ---: | ---: | --- |
| handbook-alpha.pdf | 7 | 3 | 540 | yes |
| handbook-beta.pdf | 7 | 3 | 540 | yes |
| handbook-gamma.pdf | 7 | 3 | 540 | yes |
| handbook-epsilon.pdf | 7 | 3 | 540 | yes |
| travel-manual.pdf | 6 | 3 | 540 | yes |

## Case Results

| Case | Type | Pass | Abstain | Doc Hit | Page Hit | Answer Hit | Time (ms) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| qa_remote_alpha | qa | yes | no | yes | yes | yes | 7924 |
| qa_badge_gamma | qa | yes | no | yes | yes | yes | 5267 |
| compare_remote_no_material_difference_2way | compare | yes | no | yes | yes | yes | 337 |
| compare_remote_no_material_difference_3way | compare | yes | no | yes | yes | yes | 385 |
| compare_remote_numeric_conflict | compare | yes | no | yes | yes | yes | 41932 |
| compare_remote_mixed_duplicate_conflict | compare | yes | no | yes | yes | yes | 62476 |
| qa_satellite_stipend_abstain | qa | yes | yes | yes | yes | yes | 181 |
| compare_remote_single_doc_abstain | compare | yes | yes | yes | yes | yes | 289 |
