import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

/**
 * Section-scoped report writer. Each bench mode owns one "# <title>" section
 * in the shared results file; rewriting a section leaves the others intact,
 * so a sweep re-run never clobbers the ptt table and vice versa.
 */
export async function upsertReportSection(
  path: string,
  header: string,
  lines: string[],
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const existing = await readFile(path, "utf8").catch(() => "");
  const section = [header, "", ...lines, ""].join("\n");
  const remainder = removeSection(existing, header).trimEnd();
  const content = remainder ? `${remainder}\n\n${section}` : section;
  await writeFile(path, content);
}

function removeSection(content: string, header: string): string {
  const lines = content.split("\n");
  const start = lines.findIndex((line) => line.trim() === header);
  if (start === -1) {
    return content;
  }
  let end = start + 1;
  while (end < lines.length && !lines[end]!.startsWith("# ")) {
    end += 1;
  }
  return [...lines.slice(0, start), ...lines.slice(end)].join("\n");
}
