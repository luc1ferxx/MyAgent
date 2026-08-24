// Deterministic fixture PDFs for the DocCompare verification harness.
//
// Built by hand with uncompressed text streams rather than taken from a real
// document, for one reason that the harness depends on completely: every string
// and the page it sits on is known here, so a citation's page number can be
// checked against ground truth instead of merely checked for being present.
//
// The two contracts agree on structure and disagree on two numbers. The liability
// clause is deliberately on page 2, not page 1, so a citation that always says
// "page 1" fails. The governing-law clause on page 3 differs too, and is never
// asked about -- it is there so a retrieval system has a plausible wrong answer
// available.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const escapeText = (text) => text.replace(/([\\()])/g, "\\$1");

const buildContentStream = (lines) =>
  [
    "BT",
    "/F1 12 Tf",
    "72 720 Td",
    "16 TL",
    ...lines.map((line) => `(${escapeText(line)}) Tj T*`),
    "ET",
  ].join("\n");

export const buildPdf = (pages) => {
  const objects = [];
  const pageObjectNumbers = pages.map((_, index) => 3 + index * 2);
  const fontObjectNumber = 3 + pages.length * 2;

  objects.push("<< /Type /Catalog /Pages 2 0 R >>");
  objects.push(
    `<< /Type /Pages /Kids [${pageObjectNumbers
      .map((number) => `${number} 0 R`)
      .join(" ")}] /Count ${pages.length} >>`
  );

  for (const [index, lines] of pages.entries()) {
    const contentObjectNumber = pageObjectNumbers[index] + 1;
    objects.push(
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] ` +
        `/Resources << /Font << /F1 ${fontObjectNumber} 0 R >> >> ` +
        `/Contents ${contentObjectNumber} 0 R >>`
    );

    const contentStream = buildContentStream(lines);
    objects.push(
      `<< /Length ${Buffer.byteLength(contentStream)} >>\nstream\n${contentStream}\nendstream`
    );
  }

  objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");

  let pdf = "%PDF-1.4\n";
  const offsets = [];

  for (const [index, body] of objects.entries()) {
    offsets.push(Buffer.byteLength(pdf));
    pdf += `${index + 1} 0 obj\n${body}\nendobj\n`;
  }

  const startXref = Buffer.byteLength(pdf);
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;

  for (const offset of offsets) {
    pdf += `${String(offset).padStart(10, "0")} 00000 n \n`;
  }

  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${startXref}\n%%EOF\n`;

  return Buffer.from(pdf, "latin1");
};

const vendorPages = ({ vendor, liabilityMonths, noticeDays, state }) => [
  [
    `VENDOR ${vendor} MASTER SERVICES AGREEMENT`,
    "Section 1. Scope of Services.",
    `Vendor ${vendor} shall provide cloud hosting and support services.`,
  ],
  [
    "Section 7. Limitation of Liability.",
    `The total liability of Vendor ${vendor} shall not exceed the fees paid`,
    `in the ${liabilityMonths} months preceding the claim.`,
    "Section 8. Termination.",
    `Either party may terminate this agreement on ${noticeDays} days`,
    "written notice to the other party.",
  ],
  ["Section 12. Governing Law.", `This agreement is governed by the laws of ${state}.`],
];

// GROUND TRUTH. The harness asserts against this rather than against strings
// duplicated in its own checks, so the fixtures and the expectations cannot drift
// apart.
export const DOCCOMPARE_FIXTURES = Object.freeze({
  vendorA: Object.freeze({
    fileName: "vendor-a.pdf",
    docId: "verify-vendor-a",
    pageCount: 3,
    liabilityPage: 2,
    liabilityValue: "twelve (12)",
    liabilityNumber: "12",
    // Both surface forms are recorded because a model may write either, and the
    // harness has to accept a correct answer in whichever form it arrives.
    liabilityWord: "twelve",
    noticeValue: "thirty (30)",
    governingLawPage: 3,
    pages: vendorPages({
      vendor: "A",
      liabilityMonths: "twelve (12)",
      noticeDays: "thirty (30)",
      state: "Delaware",
    }),
  }),
  vendorB: Object.freeze({
    fileName: "vendor-b.pdf",
    docId: "verify-vendor-b",
    pageCount: 3,
    liabilityPage: 2,
    liabilityValue: "six (6)",
    liabilityNumber: "6",
    liabilityWord: "six",
    noticeValue: "ninety (90)",
    governingLawPage: 3,
    pages: vendorPages({
      vendor: "B",
      liabilityMonths: "six (6)",
      noticeDays: "ninety (90)",
      state: "New York",
    }),
  }),
  // A pair that is genuinely identical apart from the file name, used as a
  // negative control: a system that invents differences to look useful will
  // report some here, and must not.
  twinLeft: Object.freeze({
    fileName: "policy-v1.pdf",
    docId: "verify-twin-left",
    pageCount: 2,
    pages: [
      ["REMOTE WORK POLICY", "Section 1. Eligibility."],
      [
        "Section 2. Allowance.",
        "Employees may work remotely two (2) days per week",
        "with manager approval.",
      ],
    ],
  }),
  twinRight: Object.freeze({
    fileName: "policy-v2.pdf",
    docId: "verify-twin-right",
    pageCount: 2,
    pages: [
      ["REMOTE WORK POLICY", "Section 1. Eligibility."],
      [
        "Section 2. Allowance.",
        "Employees may work remotely two (2) days per week",
        "with manager approval.",
      ],
    ],
  }),
});

export const writeDocCompareFixtures = async (outputDirectory) => {
  await mkdir(outputDirectory, { recursive: true });

  const written = [];

  for (const fixture of Object.values(DOCCOMPARE_FIXTURES)) {
    const filePath = path.join(outputDirectory, fixture.fileName);
    await writeFile(filePath, buildPdf(fixture.pages));
    written.push({ ...fixture, filePath });
  }

  return written;
};

// Also runnable directly, so the PDFs can be inspected by hand or fed to the CLI.
if (process.argv[1] && process.argv[1].endsWith("build-doccompare-fixtures.mjs")) {
  const outputDirectory = process.argv[2];

  if (!outputDirectory) {
    console.error("Usage: node evaluation/build-doccompare-fixtures.mjs <output-directory>");
    process.exitCode = 1;
  } else {
    const written = await writeDocCompareFixtures(outputDirectory);
    console.log(
      `Wrote ${written.length} fixture PDFs to ${outputDirectory}:\n` +
        written.map((fixture) => `  ${fixture.fileName} (${fixture.pageCount} pages)`).join("\n")
    );
  }
}
