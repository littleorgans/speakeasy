#!/usr/bin/env python3
"""Run local saved WAV accuracy only; no timing or network operations."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from prepare import HERE, HEX_REV, RUNTIME_REV, digest

THREAD = Path('/Users/alphab/.transport-matters-preview/threads/tm-ui--e278bb71/known-words-fix--3eac27aa')
DEFAULT_SCORER = Path('/Users/alphab/.transport-matters-preview/forks/tm-ui--e278bb71/known-words-fix--3eac27aa/known-words-fix--8588000c/experiments/known-words/shared/scorer.mjs')
MODEL = Path('/Users/alphab/Library/Application Support/hex/models/parakeet-unified-en-0.6b-Q8_0.gguf')
MODEL_SHA = '4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795'
GROK_IDS = ['p013','p041','p083','p097','p104','p123','p129','p134','p139','p142','p149']
SCORER_FILES = {
    'experiments/known-words/shared/scorer.mjs': '7a5a93ef226ba5490c6b2e2510c98d6cce30daf151ad01a1c23e5289c28913f3',
    'experiments/known-words/shared/io.mjs': '47a502d00855257656522aca358e003f53d17acb9ab48a00ae660f3dd1764d7c',
    'packages/speech-io/src/bench/normalize.ts': 'e0aba9d2cfb48c2123969533609c875782b37a69d80507377345b10beb0b14db',
    'packages/speech-io/src/bench/transcript.ts': '9f1605dcf471d5d24e26b08968d4e72a6c9801b9c35d3753464d3a769842e438',
    'packages/speech-io/src/rewrite/numbers.ts': '75d285055a71f760ba5e98994e03a746aa3b8d7e0ff783c830bf467dd66b8c40',
}


def canonical(index_path):
    index = json.loads(index_path.read_text())
    takes = []
    for descriptor in index['takeFiles']:
        path = index_path.parent.parent / descriptor['path']
        assert path.stat().st_size == descriptor['bytes']
        assert digest(path) == descriptor['sha256']
        take = json.loads(path.read_text())
        assert take['takeId'] == descriptor['takeId']
        takes.append(take)
    assert len(takes) == 160 and len({t['takeId'] for t in takes}) == 160
    assert sum(t['partition'] == 'selected' for t in takes) == 150
    assert sum(t['partition'] == 'today' for t in takes) == 10
    assert [t['promptId'] for t in takes if t['partition'] == 'selected'
            and re.search(r'\bgrok\b', t['reference'], re.I)] == GROK_IDS
    return takes


def validate_audio(takes):
    for take in takes:
        assert take['sampleRate'] == 16000
        assert digest(Path(take['audioPath'])) == take['audioSha256']
        assert digest(Path(take['sidecarPath'])) == take['sidecarSha256']


def validate_settings():
    path = Path('/Users/alphab/Library/Application Support/hex/settings.json')
    settings = json.loads(path.read_text())
    assert settings['transcription'] == {'model':'parakeet_unified_en','language':'en'}
    for flag in ['vocabulary_biasing','spoken_formatting','spoken_symbols','filler_words']:
        assert not settings['labs'][flag]
    processing = settings['dictation_processing']
    assert not processing['modes']
    default = processing['default_mode']
    assert not default['replacements'] and not default['transformations']
    assert not default['post_processing']['enabled']
    return digest(path)


def reference_metadata(takes):
    groups = {}
    for take in takes:
        key = (take['referenceStatus'],take['referenceSource'])
        groups.setdefault(key,[]).append(take['takeId'])
    return [{'referenceStatus':status,'referenceSource':source,'takeIds':ids}
            for (status,source),ids in groups.items()]


def run(args):
    assert args.work.parent == Path('/private/tmp') and (args.work / '.hex-grok-repro').exists()
    identity = json.loads((args.work / 'build-identity.json').read_text())
    assert identity['runtimeCommit'] == RUNTIME_REV and identity['hexCommit'] == HEX_REV
    executable = Path(identity['executable'])
    assert digest(executable) == identity['executableSha256']
    assert digest(HERE / 'runner.rs') == identity['runnerSourceSha256']
    assert digest(args.model) == MODEL_SHA
    settings_hash = validate_settings()
    scorer_root = args.scorer.parents[3]
    for rel, expected in SCORER_FILES.items():
        assert digest(scorer_root / rel) == expected
    partition = 'today' if args.cohort == 'today' else 'selected'
    takes = [t for t in canonical(args.index) if t['partition'] == partition]
    if args.cohort == 'grok':
        takes = [t for t in takes if t['promptId'] in GROK_IDS]
    if args.cohort == 'today':
        assert len(takes) == 10
        assert all(t['reference'] == 'launch 1 grok and 1 claude' for t in takes)
    validate_audio(takes)
    # Fail before executing if the destination already exists.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.work / f'{args.cohort}-{args.mode}-manifest.jsonl'
    manifest.write_text(''.join(json.dumps({'takeId':t['takeId'],'audioPath':t['audioPath']})+'\n' for t in takes))
    takefile = args.work / f'{args.cohort}-{args.mode}-takes.json'
    takefile.write_text(json.dumps([{k:t[k] for k in ['takeId','promptId','partition','reference',
        'rawReference','savedReference','referenceSource','referenceStatus']} for t in takes]))
    raw = args.work / f'{args.cohort}-{args.mode}-raw.jsonl'
    log = args.work / f'{args.cohort}-{args.mode}-inference.log'
    # Explicit, minimal child environment excludes caller runtime overrides and
    # credentials. No temporary directory environment variable is changed.
    env = {'PATH':'/opt/homebrew/bin:/usr/bin:/bin', 'HOME':str(args.work/'home')}
    with raw.open('w') as out, log.open('w') as err:
        status = subprocess.run([str(executable), str(args.model), str(manifest), args.mode],
                                env=env, stdout=out, stderr=err).returncode
    print(f'accuracy recognizer exit {status}', flush=True)
    (args.output / 'execution.json').write_text(json.dumps({'exit':status, 'expectedTakes':len(takes),
        'cohort':args.cohort, 'mode':args.mode, 'build':identity}, indent=2)+'\n')
    if status:
        raise SystemExit(status)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    records = [r for r in rows if 'takeId' in r]
    assert [r['takeId'] for r in records] == [t['takeId'] for t in takes]
    assert all(r['inputSamples'] == t['sampleCount'] for r,t in zip(records,takes))
    with (args.work / 'score.log').open('w') as log_stream:
        status = subprocess.run([args.node, str(HERE/'evaluate.mjs'), str(args.scorer),
                                str(takefile), str(raw), str(args.output/'scores.json')],
                                stdout=log_stream, stderr=log_stream, env=env).returncode
    print(f'retained scorer exit {status}', flush=True)
    if status:
        raise SystemExit(status)
    # Compact stage encoding: an omitted recognizerText equals rawRecognizer;
    # an omitted hexFinal equals recognizerText. Post processing is disabled.
    compact = []
    for row in records:
        r = dict(row)
        if r['hexFinal'] == r['recognizerText']:
            del r['hexFinal']
        if r['recognizerText'] == r['rawRecognizer']:
            del r['recognizerText']
        compact.append(r)
    (args.output/'outputs.jsonl').write_text(''.join(json.dumps(r, separators=(',',':'),ensure_ascii=False)+'\n' for r in compact))
    # Verify inputs were not changed, and record corpus identities, not copies.
    validate_audio(takes)
    assert validate_settings() == settings_hash
    provenance = {'indexPath':str(args.index), 'indexSha256':digest(args.index),
        'modelSha256':MODEL_SHA, 'scorerFiles':SCORER_FILES, 'scorerPath':str(args.scorer),
        'runtime':rows[0], 'settingsSha256':settings_hash,
        'cohortTakeIds':[t['takeId'] for t in takes],
        'audioHashes':{t['takeId']:t['audioSha256'] for t in takes},
        'referenceMetadata':reference_metadata(takes),
        'stageEncoding':'omitted recognizerText = rawRecognizer; omitted hexFinal = recognizerText; postProcessed null = disabled',
        'timing':'not measured; runtime timing fields discarded'}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'complete: {len(takes)} canonical takes; inputs reverified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--cohort',choices=['grok','selected','today'],default='grok')
    parser.add_argument('--mode',choices=['stream','offline'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=Path,default=THREAD/'takes/index.json')
    parser.add_argument('--scorer',type=Path,default=DEFAULT_SCORER)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--node',default='/Users/alphab/.vite-plus/bin/node')
    run(parser.parse_args())
