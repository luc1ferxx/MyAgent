import React from "react";
import { Spin } from "antd";

const RenderQA = (props) => {
  const {
    conversation,
    activeTurnIndex,
    isLoading,
    selectedSource,
    onSelectSource,
    onSelectTurn,
  } = props;

  if (!conversation?.length && !isLoading) {
    return (
      <div className="archive-empty-log">
        <div className="archive-empty-mark">No conversation yet</div>
        <div>Upload documents on the left, then ask a question to begin.</div>
      </div>
    );
  }

  return (
    <div className="archive-log">
      {conversation?.map((each, index) => {
        return (
          <article
            key={index}
            className={`archive-entry ${
              activeTurnIndex === index ? "is-active" : ""
            }`}
            onClick={() => onSelectTurn?.(index)}
          >
            <div className="archive-entry-eyebrow">Prompt {index + 1}</div>

            <div className="archive-question">
              <div className="archive-question-label">You</div>
              <div className="archive-question-text">{each.question}</div>
            </div>

            <section className="archive-response">
              <div className="archive-response-section">
                <div className="archive-answer-label-wrap">
                  <div className="archive-answer-label">Document answer</div>
                  {each.answer?.ragMemoryApplied ? (
                    <span className="archive-answer-chip">Memory</span>
                  ) : null}
                </div>

                <div className="archive-answer-text">{each.answer.ragAnswer}</div>

                {each.answer.ragSources?.length > 0 && (
                  <div className="archive-source-list">
                    <div className="archive-source-section-label">Citations</div>

                    {each.answer.ragSources.map((source) => (
                      <button
                        key={`${source.docId}-${source.chunkIndex}-${source.rank}`}
                        type="button"
                        className={`archive-source-item ${
                          selectedSource?.docId === source.docId &&
                          selectedSource?.chunkIndex === source.chunkIndex
                            ? "is-selected"
                            : ""
                        }`}
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelectTurn?.(index);
                          onSelectSource?.(source);
                        }}
                      >
                        <div className="archive-source-head">
                          <span>{source.fileName}</span>
                          <span>
                            {source.pageNumber ? `Page ${source.pageNumber}` : ""}
                          </span>
                        </div>
                        <div className="archive-source-copy">{source.excerpt}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="archive-response-divider" />

              <div className="archive-response-section archive-response-section-secondary">
                <div className="archive-answer-label-wrap">
                  <div className="archive-answer-label">Web answer</div>
                  {each.answer?.errors?.mcp ? (
                    <span className="archive-answer-chip is-muted">Fallback</span>
                  ) : null}
                </div>

                <div className="archive-answer-text">{each.answer.mcpAnswer}</div>
              </div>
            </section>
          </article>
        );
      })}

      {isLoading && (
        <div className="archive-loading">
          <Spin size="large" />
          <div className="archive-loading-copy">Generating an answer...</div>
        </div>
      )}
    </div>
  );
};

export default RenderQA;
