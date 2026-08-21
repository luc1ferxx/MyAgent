// Pure helpers for the archive MCP server (server/archive-mcp-server.js).
//
// Everything here is transport-free and provider-free so it can be unit tested
// without OpenAI, a vector store, or PostgreSQL. The MCP wiring file owns the
// side effects; this file owns the shapes.

const EXCERPT_LIMIT = 220;

const normalizeText = (value) => String(value ?? "").replace(/\s+/g, " ").trim();

const toArray = (value) => (Array.isArray(value) ? value : []);

const toFiniteNumber = (value) =>
  Number.isFinite(Number(value)) ? Number(value) : null;

/**
 * Projects a `buildCitation` record (server/rag/citations.js) onto the wire.
 * Field names are preserved verbatim so an MCP consumer sees the same citation
 * contract the web workbench renders.
 */
export const formatCitation = (citation = {}) => ({
  rank: toFiniteNumber(citation.rank),
  score: toFiniteNumber(citation.score),
  docId: citation.docId ?? null,
  fileName: citation.fileName ?? "Unknown document",
  filePath: citation.filePath ?? "",
  pageNumber: toFiniteNumber(citation.pageNumber),
  chunkIndex: toFiniteNumber(citation.chunkIndex),
  excerpt: normalizeText(citation.excerpt).slice(0, EXCERPT_LIMIT),
  sectionHeading: citation.sectionHeading ?? null,
});

export const formatCitations = (citations) =>
  toArray(citations).map((citation) => formatCitation(citation));

/**
 * Projects `retrievedContexts` entries (server/rag/answer-writer.js). These
 * carry the full chunk text and are matched to citations by
 * docId + chunkIndex + pageNumber.
 */
export const formatEvidence = (retrievedContexts) =>
  toArray(retrievedContexts).map((context = {}) => ({
    rank: toFiniteNumber(context.rank),
    docId: context.docId ?? null,
    fileName: context.fileName ?? "Unknown document",
    pageNumber: toFiniteNumber(context.pageNumber),
    chunkIndex: toFiniteNumber(context.chunkIndex),
    sectionHeading: context.sectionHeading ?? null,
    text: String(context.text ?? ""),
  }));

/**
 * Shapes a planner-free `chat()` result for MCP.
 *
 * `abstained` and `abstainReason` come first on purpose: when the confidence
 * gate declines to answer, that has to be the first thing the calling model
 * reads. On abstention we deliberately return no citations, so there is nothing
 * for the model to dress up as evidence.
 */
export const formatAskResult = (response = {}) => {
  const abstained = Boolean(response.abstained);
  const abstainReason = abstained
    ? normalizeText(response.abstainReason ?? response.text) || "No reason provided."
    : null;

  return {
    abstained,
    abstainReason,
    answer: abstained ? "" : String(response.text ?? ""),
    citations: abstained ? [] : formatCitations(response.citations),
    evidence: abstained ? [] : formatEvidence(response.retrievedContexts),
    resolvedQuestion: response.resolvedQuery ?? null,
    comparison: response.comparisonAnalysisSummary ?? null,
  };
};

/**
 * Errors are reported in the same shape as an abstention rather than thrown, so
 * a failing archive degrades into "I have no evidence" instead of leaving the
 * calling model to invent an answer.
 */
export const formatToolError = (error) => ({
  abstained: true,
  abstainReason: normalizeText(error?.message) || "The archive request failed.",
  answer: "",
  citations: [],
  evidence: [],
  error: normalizeText(error?.message) || "The archive request failed.",
});

export const formatDocumentList = (documents) =>
  toArray(documents).map((document = {}) => ({
    docId: document.docId ?? null,
    fileName: document.fileName ?? "Unknown document",
    pageCount: toFiniteNumber(document.pageCount),
    chunkCount: toFiniteNumber(document.chunkCount),
    uploadedAt: document.uploadedAt ?? null,
  }));

export const parseDocIdList = (docIds) =>
  toArray(docIds)
    .map((docId) => normalizeText(docId))
    .filter((docId) => docId.length > 0);

/**
 * The planner-free `chat()` entry throws 404 when no docId is supplied
 * (server/rag/index.js:244-248), unlike `POST /chat` which resolves workspace
 * scope through the agent. So an omitted `docIds` has to be widened to "every
 * document in scope" here.
 */
export const resolveDocIds = ({ docIds, availableDocuments } = {}) => {
  const requested = parseDocIdList(docIds);

  if (requested.length > 0) {
    return requested;
  }

  return parseDocIdList(
    toArray(availableDocuments).map((document) => document?.docId)
  );
};

export const buildEmptyArchiveResult = () => ({
  abstained: true,
  abstainReason:
    "The archive has no documents in scope. Upload a PDF in the workbench, then call archive_refresh.",
  answer: "",
  citations: [],
  evidence: [],
});

export const toMcpTextContent = (payload) => ({
  content: [
    {
      type: "text",
      text: JSON.stringify(payload, null, 2),
    },
  ],
});
