---
description: Ask the PDF archive a question and get a page-cited answer
agent: archive-rag
---

Answer this question from the PDF archive: $ARGUMENTS

Use `archive_ask`. Call `archive_list_documents` first if you need to identify
which documents are relevant, or if the question names specific documents to
scope or compare.

Report, in this order:

1. The answer.
2. The citations behind it — `fileName`, page number, and the excerpt for each.
3. If `abstained` is true, report `abstainReason` verbatim instead of an answer,
   and suggest either rephrasing or uploading the missing document in the web
   workbench followed by `archive_refresh`.

Do not supply an answer from your own knowledge when the archive abstains, and do
not cite any page that is not in the returned citations.
