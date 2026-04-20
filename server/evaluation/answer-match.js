const UNIT_NUMBER_WORDS = new Map([
  ["zero", 0],
  ["one", 1],
  ["two", 2],
  ["three", 3],
  ["four", 4],
  ["five", 5],
  ["six", 6],
  ["seven", 7],
  ["eight", 8],
  ["nine", 9],
]);

const TEEN_NUMBER_WORDS = new Map([
  ["ten", 10],
  ["eleven", 11],
  ["twelve", 12],
  ["thirteen", 13],
  ["fourteen", 14],
  ["fifteen", 15],
  ["sixteen", 16],
  ["seventeen", 17],
  ["eighteen", 18],
  ["nineteen", 19],
]);

const TENS_NUMBER_WORDS = new Map([
  ["twenty", 20],
  ["thirty", 30],
  ["forty", 40],
  ["fifty", 50],
  ["sixty", 60],
  ["seventy", 70],
  ["eighty", 80],
  ["ninety", 90],
]);

const SIMPLE_NUMBER_WORDS = new Set([
  ...UNIT_NUMBER_WORDS.keys(),
  ...TEEN_NUMBER_WORDS.keys(),
  ...TENS_NUMBER_WORDS.keys(),
]);

const NUMBER_WORD_PATTERN = new RegExp(
  `\\b(?:${[...SIMPLE_NUMBER_WORDS].join("|")})(?:[-\\s]+(?:${[
    ...UNIT_NUMBER_WORDS.keys(),
  ].join("|")}))?\\b`,
  "g"
);

const normalizeNumberWordPhrase = (phrase) => {
  const normalizedPhrase = phrase.toLowerCase().replace(/-/g, " ").trim();
  const parts = normalizedPhrase.split(/\s+/);

  if (parts.length === 1) {
    if (UNIT_NUMBER_WORDS.has(parts[0])) {
      return String(UNIT_NUMBER_WORDS.get(parts[0]));
    }

    if (TEEN_NUMBER_WORDS.has(parts[0])) {
      return String(TEEN_NUMBER_WORDS.get(parts[0]));
    }

    if (TENS_NUMBER_WORDS.has(parts[0])) {
      return String(TENS_NUMBER_WORDS.get(parts[0]));
    }

    return normalizedPhrase;
  }

  if (
    parts.length === 2 &&
    TENS_NUMBER_WORDS.has(parts[0]) &&
    UNIT_NUMBER_WORDS.has(parts[1])
  ) {
    return String(TENS_NUMBER_WORDS.get(parts[0]) + UNIT_NUMBER_WORDS.get(parts[1]));
  }

  return normalizedPhrase;
};

const normalizeNumberWords = (text) =>
  text.replace(NUMBER_WORD_PATTERN, (match) => normalizeNumberWordPhrase(match));

export const normalizeAnswerForMatch = (text) =>
  normalizeNumberWords(
    String(text ?? "")
      .toLowerCase()
      .replace(/(\d),(\d)/g, "$1$2")
  )
    .replace(/\s+/g, " ")
    .trim();

export const evaluateAnswerExpectation = ({ answer, expectedAnswerIncludes }) => {
  if (!Array.isArray(expectedAnswerIncludes) || expectedAnswerIncludes.length === 0) {
    return true;
  }

  const normalizedAnswer = normalizeAnswerForMatch(answer);

  return expectedAnswerIncludes.every((expectedFragment) =>
    normalizedAnswer.includes(normalizeAnswerForMatch(expectedFragment))
  );
};
