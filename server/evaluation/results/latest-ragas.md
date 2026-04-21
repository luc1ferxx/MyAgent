# Ragas Evaluation

- Created: `2026-04-21T09:40:22.362674+00:00`
- Input file: `C:\Users\Jxx\Desktop\agentai\server\evaluation\results\latest.json`
- Source run ID: `2026-04-21T09-12-35-365Z`
- Judge model: `gpt-4o-mini`
- Embedding model: `text-embedding-3-small`
- Eligible cases: `6` / `8`

## Metrics

| Metric | Average |
| --- | ---: |
| Answer relevancy | 0.6039 |
| Faithfulness | 0.576 |
| Context utilization | 0.6667 |
| Context precision | 0.8333 |
| Context recall | 0.8889 |

## Case Results

| Case | Type | Answer Relevancy | Faithfulness | Context Utilization | Context Precision | Context Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qa_remote_alpha | qa | 0.6127 | 1.0 | 1.0 | 1.0 | 1.0 |
| qa_badge_gamma | qa | 0.7188 | 1.0 | 1.0 | 1.0 | 1.0 |
| compare_remote_no_material_difference_2way | compare | 0.4257 | 0.0 | 0.0 | 1.0 | 0.6667 |
| compare_remote_no_material_difference_3way | compare | 0.4196 | 0.0 | 0.0 | 1.0 | 0.6667 |
| compare_remote_numeric_conflict | compare | 0.6757 | 0.7895 | 1.0 | 0.0 | 1.0 |
| compare_remote_mixed_duplicate_conflict | compare | 0.7707 | 0.6667 | 1.0 | 1.0 | 1.0 |

## Skipped Cases

- `qa_satellite_stipend_abstain`: No retrieved_contexts were captured.
- `compare_remote_single_doc_abstain`: Abstain case skipped by default.
