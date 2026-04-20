import test from "node:test";
import assert from "node:assert/strict";
import {
  evaluateAnswerExpectation,
  normalizeAnswerForMatch,
} from "../evaluation/answer-match.js";

test("answer match normalization aligns number words with digits", () => {
  assert.equal(
    normalizeAnswerForMatch("Employees may work remotely two days per week."),
    "employees may work remotely 2 days per week."
  );
  assert.equal(
    normalizeAnswerForMatch("Renew badges every fourteen months."),
    "renew badges every 14 months."
  );
  assert.equal(
    normalizeAnswerForMatch("Flights above twenty-one dollars require approval."),
    "flights above 21 dollars require approval."
  );
});

test("answer expectation matching accepts number words for numeric expectations", () => {
  assert.equal(
    evaluateAnswerExpectation({
      answer:
        "Employees may work remotely two days per week with manager approval.",
      expectedAnswerIncludes: ["2", "manager approval"],
    }),
    true
  );
  assert.equal(
    evaluateAnswerExpectation({
      answer: "Renew badges every fourteen months after the last audit.",
      expectedAnswerIncludes: ["14", "months"],
    }),
    true
  );
});
