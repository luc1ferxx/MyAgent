import React from "react";
import { Button } from "antd";
import { API_DOMAIN } from "../config";

const PdfPreview = ({ source }) => {
  if (!source?.filePath) {
    return (
      <div className="archive-preview-empty">
        <div className="archive-preview-empty-mark">No source selected</div>
        <div>Choose a citation to preview the related page.</div>
      </div>
    );
  }

  const pageNumber = source.pageNumber ?? 1;
  const previewUrl = `${API_DOMAIN}/${source.filePath}#page=${pageNumber}&view=FitH`;

  return (
    <div className="archive-preview-wrap">
      <div className="archive-preview-meta">
        <div>
          <div className="archive-preview-file">{source.fileName}</div>
          <div className="archive-preview-page">Page {pageNumber}</div>
        </div>

        <Button
          className="archive-secondary-button"
          href={previewUrl}
          target="_blank"
          rel="noreferrer"
        >
          Open full page
        </Button>
      </div>

      {source.excerpt ? (
        <p className="archive-preview-excerpt">{source.excerpt}</p>
      ) : null}

      <iframe
        className="archive-preview-frame"
        src={previewUrl}
        title={`${source.fileName} preview`}
      />
    </div>
  );
};

export default PdfPreview;
