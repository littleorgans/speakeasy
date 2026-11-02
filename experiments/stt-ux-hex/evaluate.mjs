// Use the retained scorer directly, without its removed manifest loader.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [scorerPath, takesPath, outputPath, destination] = process.argv.slice(2);
const { scoreCorpus, normalize } = await import(pathToFileURL(scorerPath));
const takes = JSON.parse(readFileSync(takesPath, 'utf8'));
const outputs = readFileSync(outputPath, 'utf8').trim().split('\n').map(JSON.parse)
  .filter(row => row.takeId);
const result = scoreCorpus(takes, outputs.map(row => ({ takeId: row.takeId, text: row.hexFinal })));
assert.equal(result.rows.length, takes.length);
const details = result.rows.map((row, i) => ({
  promptId: takes[i].promptId, takeId: row.takeId,
  normalizedText: row.normalizedText,
  wer: row.wer, grok: row.knownWords.grok,
  grokWholeWord: /\bgrok\b/.test(normalize(row.text)), exact: row.exact,
}));
// No inference timing is supplied, retained, or interpreted as latency.
delete result.metrics.all.latency;
delete result.metrics.selected.latency;
delete result.metrics.today.latency;
writeFileSync(destination, JSON.stringify({ metrics: result.metrics, details }) + '\n');
