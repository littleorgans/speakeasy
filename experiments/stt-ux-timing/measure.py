#!/usr/bin/env python3
"""Local paced inference. Child diagnostics stay in the isolated build directory."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'stt-ux-hex'))
from prepare import digest, HEX_REV, RUNTIME_REV
from run import (THREAD, DEFAULT_SCORER, MODEL, MODEL_SHA, GROK_IDS, SCORER_FILES,
                 canonical, validate_audio, validate_settings, reference_metadata)
from metrics import summarize, aggregate

NODE = '/Users/alphab/.vite-plus/bin/node'


def write(path, obj):
    path.write_text(json.dumps(obj, separators=(',',':'), ensure_ascii=False)+'\n')


class Loads:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll)
        self.thread.start()

    def observe(self):
        with self.lock:
            if self.samples is not None:
                self.samples.append([time.monotonic_ns(), *os.getloadavg()])

    def poll(self):
        while not self.stop.wait(0.05):
            self.observe()

    def begin(self):
        with self.lock:
            assert self.samples is None
            self.samples = []
        self.observe()

    def end(self):
        self.observe()
        with self.lock:
            samples, self.samples = self.samples, None
        origin = samples[0][0]
        series = [[(r[0]-origin)/1e6, *r[1:]] for r in samples]
        maximum = [max(r[i] for r in series) for i in [1,2,3]]
        return {'start':series[0][1:], 'end':series[-1][1:], 'max':maximum,
                'contended':maximum[0] > 4, 'observations':len(series),
                'requestedPollMs':50, 'wallSpanMs':series[-1][0],
                'maxPollGapMs':max(b[0]-a[0] for a,b in zip(series,series[1:])),
                'boundary':'parent receipt sample_start through sample result; includes setup/serialization'}, series

    def close(self):
        self.stop.set()
        self.thread.join()


def machine(work):
    result = {'system':platform.system(), 'release':platform.release(),
              'version':platform.version(), 'architecture':platform.machine(),
              'macOS':platform.mac_ver()[0], 'python':platform.python_version(),
              'logicalCPUs':os.cpu_count()}
    for label, command in [('chip',['sysctl','-n','machdep.cpu.brand_string']),
                           ('memoryBytes',['sysctl','-n','hw.memsize']),
                           ('physicalCPUs',['sysctl','-n','hw.physicalcpu']),
                           ('osBuild',['sw_vers','-buildVersion']),
                           ('node',[NODE,'--version'])]:
        with (work/('identity-'+label+'.log')).open('w') as err:
            p = subprocess.run(command, stdout=subprocess.PIPE, stderr=err, text=True,
                               env={'PATH':'/usr/bin:/usr/sbin:/bin'})
        assert p.returncode == 0
        result[label] = p.stdout.strip()
    return result


def frozen_panel():
    takes = canonical(THREAD/'takes/index.json')
    panel = [t for t in takes if t['partition']=='selected' and t['promptId'] in GROK_IDS]
    panel += [t for t in takes if t['partition']=='today']
    assert len(panel) == 21 and [t['promptId'] for t in panel[:11]] == GROK_IDS
    assert all(t['reference']=='launch 1 grok and 1 claude' for t in panel[11:])
    validate_audio(panel)
    return panel


def validate(work):
    build = json.loads((work/'timing-build-identity.json').read_text())
    assert build['hexCommit']==HEX_REV and build['runtimeCommit']==RUNTIME_REV
    assert digest(Path(build['executable'])) == build['executableSha256']
    assert digest(HERE/'runner.rs') == build['timingRunnerSha256']
    assert digest(MODEL) == MODEL_SHA
    for rel, expected in build['sourceHashes'].items():
        assert digest(work/rel) == expected
    for rel, expected in SCORER_FILES.items():
        assert digest(DEFAULT_SCORER.parents[3]/rel) == expected
    return build


def process(work, condition, manifest, callback):
    build = validate(work)
    env = {'PATH':'/opt/homebrew/bin:/usr/bin:/bin','HOME':str(work/'home')}
    with (work/(condition+'-timing-native.log')).open('w') as log:
        child = subprocess.Popen([build['executable'], str(MODEL), str(manifest), condition],
                                  env=env, stdout=subprocess.PIPE, stderr=log, text=True)
        try:
            for line in child.stdout:
                callback(json.loads(line))
        except BaseException:
            child.terminate()
            child.wait()
            raise
        status = child.wait()
    print(condition+' recognizer exit '+str(status), flush=True)
    if status:
        raise SystemExit(status)
    return status


def probe(args):
    build = validate(args.work)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    status = process(args.work, 'probe', args.work/'unused-manifest', rows.append)
    probes = [r for r in rows if r['kind']=='geometry_probe']
    assert [p['condition'] for p in probes]==['default','low160']
    assert all(p['accepted'] and not p['initialFull'] and not p['initialCommitted']
               and not p['initialTentative'] for p in probes)
    panel = frozen_panel()
    write(args.output/'protocol.json', {
        'frozenUtc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'geometryProbe':rows, 'probeExit':status, 'build':build,
        'machine':machine(args.work), 'settingsSha256':validate_settings(),
        'indexPath':str(THREAD/'takes/index.json'), 'indexSha256':digest(THREAD/'takes/index.json'),
        'panel':[{k:t[k] for k in ['takeId','promptId','partition','audioSha256','sampleCount']}
                 for t in panel], 'referenceMetadata':reference_metadata(panel),
        'accuracyEvidenceHashes':{str(p.relative_to(HERE.parent)):digest(p)
            for p in sorted((HERE.parent/'stt-ux-hex').rglob('*')) if p.is_file()},
        'settings':{'backend':'Metal','sessionThreads':1,'language':'en',
            'timestamps':'Segment','feedSamples':2560,'sampleRate':16000,'pace':1,
            'suffixSamples':6400,'warmupZeroSamples':160000,'commitPolicy':'Auto',
            'stablePrefixAgreementN':0,'prompt':None,'prefix':None,'vocabulary':[],
            'leftOverride':None,'fallback':'observed eligibility only, no substitution',
            'liveTextSaved':False,'loadThresholdOneMinute':4,'loadPollMs':50}})
    print('both geometries accepted; frozen all21 panel and identity retained', flush=True)


def save_snapshots(folder, snapshots):
    part, size, stream = 0, 0, None
    try:
        for snapshot in snapshots:
            line = json.dumps(snapshot, separators=(',',':'),ensure_ascii=False)+'\n'
            if stream is None or size+len(line.encode()) > 60000:
                if stream is not None:
                    stream.close()
                stream = (folder/f'snapshots-{part:03}.jsonl').open('w')
                part, size = part+1, 0
            stream.write(line)
            size += len(line.encode())
    finally:
        if stream is not None:
            stream.close()


def measure(args):
    protocol = json.loads((args.output/'protocol.json').read_text())
    assert protocol['build']==validate(args.work)
    assert protocol['settingsSha256']==validate_settings()
    panel = frozen_panel()
    assert [t['takeId'] for t in panel] == [t['takeId'] for t in protocol['panel']]
    destination = args.output/args.condition
    if args.condition=='low160':
        assert json.loads((args.output/'default/execution.json').read_text())['exit']==0
    destination.mkdir(exist_ok=False)
    manifest = args.work/(args.condition+'-timing-inputs.jsonl')
    manifest.write_text(''.join(json.dumps({'takeId':t['takeId'],'audioPath':t['audioPath']})+'\n'
                                for t in panel))
    rows, runtime = [], {}
    loads = Loads()

    def receive(row):
        kind = row['kind']
        if kind in ['load_start','warmup_start','sample_start']:
            loads.begin()
        elif kind in ['identity','warmup']:
            load, series = loads.end()
            runtime[kind] = {**row,'load':load}
            write(destination/(kind+'-load.json'), series)
        elif kind == 'sample':
            load, series = loads.end()
            folder = destination/row['takeId']
            folder.mkdir()
            snapshots, feeds = row.pop('snapshots'), row.pop('feeds')
            save_snapshots(folder,snapshots)
            write(folder/'feeds.json', feeds)
            write(folder/'load.json', series)
            row.update(load=load,timing=summarize(row,snapshots,feeds))
            write(folder/'output.json',row)
            rows.append(row)
            print(f"{args.condition}: {len(rows)}/21 retained, contended={load['contended']}",flush=True)
    try:
        status = process(args.work,args.condition,manifest,receive)
    finally:
        loads.close()
    write(destination/'execution.json',{'exit':status,'expectedTakes':len(panel),'runtime':runtime})
    assert [r['takeId'] for r in rows]==[t['takeId'] for t in panel]
    takefile, raw = args.work/(args.condition+'-scoring.json'), args.work/(args.condition+'-finals.jsonl')
    write(takefile,panel)
    raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with (args.work/(args.condition+'-scorer.log')).open('w') as log:
        score = subprocess.run([NODE,str(HERE.parent/'stt-ux-hex/evaluate.mjs'),
            str(DEFAULT_SCORER),str(takefile),str(raw),str(destination/'scores.json')],
            stdout=log,stderr=log,env={'PATH':'/usr/bin:/bin','HOME':str(args.work/'home')})
    print('retained scorer exit '+str(score.returncode),flush=True)
    if score.returncode:
        raise SystemExit(score.returncode)
    write(destination/'summary.json', {name:aggregate(subset) for name,subset in [
        ('historical11',rows[:11]),('today10',rows[11:]),('all21',rows)]})
    validate_audio(panel)
    assert validate_settings()==protocol['settingsSha256']
    assert digest(MODEL)==MODEL_SHA
    for rel, expected in protocol['accuracyEvidenceHashes'].items():
        assert digest(HERE.parent/rel)==expected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--condition',choices=['probe','default','low160'],required=True)
    args = parser.parse_args()
    try:
        probe(args) if args.condition=='probe' else measure(args)
    except Exception:
        import traceback
        (args.work/'timing-python-error.log').write_text(traceback.format_exc())
        print('harness failure; diagnostics retained in isolated temp',flush=True)
        raise SystemExit(1)
