const pptxgen = require("/Users/jasonyan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.subject = "Citation-graph retrieval benchmark";
pptx.title = "Using Citation Graphs for Novelty-Oriented Retrieval";
pptx.company = "OpenAI";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });

const slide = pptx.addSlide();
slide.background = { color: "F7F8F3" };

const C = {
  ink: "17212B",
  muted: "58636F",
  line: "CDD4D9",
  blue: "1F6F8B",
  green: "2F7D32",
  red: "A14632",
  panel: "FFFFFF",
};

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    margin: 0.04,
    breakLine: false,
    fit: "shrink",
    fontFace: opts.fontFace || "Aptos",
    fontSize: opts.fontSize || 14,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    valign: opts.valign || "top",
    align: opts.align || "left",
    ...opts,
  });
}

function addPanel(slide, x, y, w, h, color, label, body) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.04,
    fill: { color: C.panel },
    line: { color: C.line, width: 1 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w,
    h: 0.14,
    fill: { color },
    line: { color },
  });
  addText(slide, label, x + 0.18, y + 0.28, w - 0.36, 0.48, {
    fontSize: 16,
    bold: true,
    color,
  });
  addText(slide, body, x + 0.18, y + 0.9, w - 0.36, h - 1.08, {
    fontSize: 12.5,
    color: C.ink,
    breakLine: false,
    fit: "shrink",
    paraSpaceAfterPt: 5,
  });
}

addText(
  slide,
  "Citation-Graph Benchmark: From Retrieval Labels to Novelty Evidence",
  0.48,
  0.28,
  9.2,
  0.46,
  { fontFace: "Aptos Display", fontSize: 22, bold: true }
);
addText(
  slide,
  "Goal: evaluate whether a system can find the prior work that matters for judging what a target paper actually adds.",
  0.5,
  0.78,
  10.85,
  0.34,
  { fontSize: 11.8, color: C.muted }
);

addText(slide, "Target paper", 0.7, 1.32, 1.5, 0.28, {
  fontSize: 11,
  bold: true,
  color: C.blue,
  align: "center",
});
slide.addShape(pptx.ShapeType.roundRect, {
  x: 0.67,
  y: 1.62,
  w: 1.55,
  h: 0.5,
  rectRadius: 0.04,
  fill: { color: "E7F0F4" },
  line: { color: C.blue, width: 1 },
});
addText(slide, "paper + claims", 0.82, 1.78, 1.24, 0.18, {
  fontSize: 9.5,
  color: C.blue,
  align: "center",
});
slide.addShape(pptx.ShapeType.line, {
  x: 2.4,
  y: 1.87,
  w: 1.0,
  h: 0,
  line: { color: C.muted, width: 1.5, beginArrowType: "none", endArrowType: "triangle" },
});
addText(slide, "query view", 3.5, 1.32, 1.5, 0.28, {
  fontSize: 11,
  bold: true,
  color: C.green,
  align: "center",
});
slide.addShape(pptx.ShapeType.roundRect, {
  x: 3.46,
  y: 1.62,
  w: 1.55,
  h: 0.5,
  rectRadius: 0.04,
  fill: { color: "E9F3E9" },
  line: { color: C.green, width: 1 },
});
addText(slide, "user-like ask", 3.6, 1.78, 1.25, 0.18, {
  fontSize: 9.5,
  color: C.green,
  align: "center",
});
slide.addShape(pptx.ShapeType.line, {
  x: 5.2,
  y: 1.87,
  w: 1.0,
  h: 0,
  line: { color: C.muted, width: 1.5, beginArrowType: "none", endArrowType: "triangle" },
});
addText(slide, "Cited prior work", 6.3, 1.32, 1.72, 0.28, {
  fontSize: 11,
  bold: true,
  color: C.red,
  align: "center",
});
slide.addShape(pptx.ShapeType.roundRect, {
  x: 6.38,
  y: 1.62,
  w: 1.55,
  h: 0.5,
  rectRadius: 0.04,
  fill: { color: "F6ECE8" },
  line: { color: C.red, width: 1 },
});
addText(slide, "gold support", 6.53, 1.78, 1.25, 0.18, {
  fontSize: 9.5,
  color: C.red,
  align: "center",
});
slide.addShape(pptx.ShapeType.line, {
  x: 8.15,
  y: 1.87,
  w: 1.0,
  h: 0,
  line: { color: C.muted, width: 1.5, beginArrowType: "none", endArrowType: "triangle" },
});
addText(slide, "Novelty label", 9.25, 1.32, 1.7, 0.28, {
  fontSize: 11,
  bold: true,
  color: C.ink,
  align: "center",
});
slide.addShape(pptx.ShapeType.roundRect, {
  x: 9.32,
  y: 1.62,
  w: 1.7,
  h: 0.5,
  rectRadius: 0.04,
  fill: { color: "FDF4D7" },
  line: { color: "B48B1C", width: 1 },
});
addText(slide, "added info", 9.48, 1.78, 1.38, 0.18, {
  fontSize: 9.5,
  color: "6D5312",
  align: "center",
});

addPanel(
  slide,
  0.55,
  2.55,
  3.95,
  3.95,
  C.blue,
  "1. Why citation graph?",
  "Citations are author-supplied evidence of which prior work the paper is positioning against, so they are less circular than using embedding similarity and then asking an LLM to say “good / bad.”\n\nProcess it as: target paper -> cited candidate pool -> gold prior support ACUs -> label the target’s added information relative to that support.\n\nSimilarity/LLM labels still help rank candidates, but citations anchor the benchmark in real scholarly comparison behavior."
);

addPanel(
  slide,
  4.7,
  2.55,
  3.95,
  3.95,
  C.green,
  "2. How to make queries?",
  "Create several query views from the same target paper: title/abstract, extracted contribution claims, dataset ACUs, and an LLM-rephrased user question.\n\nUse the LLM rephrase only as one query condition, with a prompt such as: “What prior dataset or benchmark should I compare against for this contribution?”\n\nReport whether retrieval is robust across query styles, not just optimized for one phrasing."
);

addPanel(
  slide,
  8.85,
  2.55,
  3.95,
  3.95,
  C.red,
  "3. Non-novel work?",
  "Citation pairs alone will under-sample zero-novelty cases because published papers usually claim some contribution.\n\nAdd controls: near-duplicate/repackaging examples, dataset extensions with only format/language splits changed, and hard negatives from highly similar uncited papers.\n\nCalibrate labels into bands: repackaging, incremental, substantial. Then sample deliberately so the benchmark covers the full novelty distribution."
);

addText(
  slide,
  "Evaluation framing: retrieve the right prior support first, then score how much of the target’s claims remain unsupported by that prior work.",
  0.58,
  6.85,
  12.2,
  0.28,
  { fontSize: 11.2, bold: true, color: C.ink, align: "center" }
);

warnIfSlideHasOverlaps(slide, pptx, { ignoreDecorativeShapes: true });
warnIfSlideElementsOutOfBounds(slide, pptx);

pptx.writeFile({ fileName: "citation_graph_benchmark_slide.pptx" });
