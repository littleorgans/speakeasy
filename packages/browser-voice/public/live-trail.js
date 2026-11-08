/**
 * The revision trail of a take's live words: each update the model sent,
 * timed from the press, with the words it changed against the update before.
 */

/**
 * Word level difference: runs of same, inserted, removed and replaced words.
 * Case and punctuation alone are no change ("two" then "two,"); same runs
 * show the newer form. Strict compares words exactly, so a case or
 * punctuation fix is a change, and has no growth rule: both texts are whole.
 */
export function diffWords(before, after, { strict = false } = {}) {
  // A word still being written ("eight, n" then "eight, nine") is growth: compare without its piece.
  const prior = String(before ?? "").trim();
  const next = String(after ?? "").trim();
  if (!strict && prior && next.length > prior.length && next.startsWith(prior) && /[\p{L}\p{N}']/u.test(next[prior.length])) {
    before = prior.slice(0, Math.max(0, prior.search(/\S+$/)));
  }
  const compared = strict ? (word) => word : key;
  const a = words(before).map(compared);
  const b = words(after).map(compared);
  const shown = words(after);
  const was = words(before);
  // Longest common subsequence table, filled from the end.
  const table = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) { ops.push(["same", shown[j]]); i++; j++; }
    else if (j < b.length && (i === a.length || table[i][j + 1] >= table[i + 1][j])) ops.push(["in", shown[j++]]);
    else ops.push(["out", was[i++]]);
  }
  // Adjacent removals and insertions form one change: removed, inserted or replaced.
  const segments = [];
  for (const [op, word] of ops) {
    let last = segments.at(-1);
    if (op === "same") {
      if (last?.kind === "same") last.words.push(word);
      else segments.push({ kind: "same", was: [], words: [word] });
      continue;
    }
    if (!last || last.kind === "same") segments.push(last = { kind: "", was: [], words: [] });
    (op === "in" ? last.words : last.was).push(word);
    last.kind = last.was.length && last.words.length ? "replaced" : last.was.length ? "removed" : "inserted";
  }
  return segments;
}

/** A correction changes or removes words already shown; adding words at the end is not one. */
export function isCorrection(segments) {
  return segments.some((segment, index) => segment.kind === "removed" || segment.kind === "replaced"
    || (segment.kind === "inserted" && segments.slice(index + 1).some((later) => later.kind === "same")));
}

/** The take's updates, each with its difference from the one before. */
export function revisions(updates) {
  let previous = "";
  return updates.map((update) => {
    const segments = diffWords(previous, update.words);
    previous = update.words;
    return { ...update, segments, correction: isCorrection(segments) };
  });
}

/** One line: "12 updates, 3 corrections". */
export function trailSummary(updates) {
  const corrections = revisions(updates).filter((revision) => revision.correction).length;
  const count = (n, one) => `${n} ${one}${n === 1 ? "" : "s"}`;
  return `${count(updates.length, "update")}, ${corrections ? count(corrections, "correction") : "no corrections"}`;
}

/**
 * List items for each update, then the final text against the last update.
 * Inserted words are ins, removed words del; a replacement is both.
 */
export function trailItems(document, updates, final, formatDuration) {
  const entries = revisions(updates).map((revision) => [`+${formatDuration(revision.atMs)}`, revision.segments]);
  if (final != null) entries.push(["Final", diffWords(updates.at(-1)?.words ?? "", final)]);
  return entries.map(([time, segments]) => {
    const item = document.createElement("li");
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = time;
    item.append(when, segmentMarks(document, segments));
    return item;
  });
}

/**
 * One span of segments: removed words as del, inserted words as ins, a
 * replacement both. `sides` shows one side only: the text before (del) or
 * after (ins). An ins whose replaced words are not shown names them in its
 * title; `className` goes on each ins.
 */
export function segmentMarks(document, segments, { sides = ["del", "ins"], className = "" } = {}) {
  const said = document.createElement("span");
  for (const segment of segments) {
    if (segment.kind === "same") { said.append(text(document, segment.words)); continue; }
    if (segment.was.length && sides.includes("del")) said.append(mark(document, "del", segment.was));
    if (segment.words.length && sides.includes("ins")) {
      const hidden = segment.was.length && !sides.includes("del");
      said.append(mark(document, "ins", segment.words, { ...(className ? { className } : {}),
        ...(hidden ? { title: segment.was.join(" ") } : {}) }));
    }
  }
  return said;
}

function words(value) { return String(value ?? "").trim().split(/\s+/).filter(Boolean); }

/** A word's comparison key: letters and digits only, lower case; punctuation alone keeps itself. */
function key(word) { return word.toLowerCase().replace(/[^\p{L}\p{N}']/gu, "") || word; }

function text(document, list) {
  const span = document.createElement("span");
  span.textContent = `${list.join(" ")} `;
  return span;
}

function mark(document, tag, list, properties = {}) {
  const element = Object.assign(document.createElement(tag), properties);
  element.textContent = list.join(" ");
  const gap = document.createElement("span");
  gap.textContent = " ";
  const group = document.createElement("span");
  group.append(element, gap);
  return group;
}
