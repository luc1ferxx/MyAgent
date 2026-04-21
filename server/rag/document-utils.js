export const buildPublicFilePath = (docId = "") => {
  const normalizedDocId = String(docId ?? "").trim();

  return normalizedDocId
    ? `documents/${encodeURIComponent(normalizedDocId)}/file`
    : "";
};
