#!/usr/bin/env python3
"""Reuse the immutable accuracy build and build only the timing observer."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import sys

HERE = Path(__file__).resolve().parent
ACCURACY = HERE.parent / 'stt-ux-hex'
sys.path.insert(0, str(ACCURACY))
import prepare as accuracy_build


def metadata(path):
    """Read GGUF metadata only, leaving tensors untouched."""
    with path.open('rb') as f:
        def unpack(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        def string():
            return f.read(unpack('Q')).decode('utf8')

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                item_kind, count = unpack('I'), unpack('Q')
                return [value(item_kind) for _ in range(count)]
            return unpack({0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i',
                           6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}[kind])

        assert f.read(4) == b'GGUF' and unpack('I') == 3
        tensors, count = unpack('Q'), unpack('Q')
        entries = {string(): value(unpack('I')) for _ in range(count)}
    return tensors, {k:v for k,v in entries.items()
                     if any(s in k for s in ['att_chunk', 'subsampling', 'hop_length', 'sample_rate'])}


def prepare(root):
    assert (root / '.hex-grok-repro').is_file()
    shutil.copyfile(HERE / 'runner.rs', root / 'runner/src/main.rs')
    env = {'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
           'HOME':str(root/'home'), 'CARGO_HOME':str(root/'cargo'),
           'CARGO_TARGET_DIR':str(root/'target'), 'CARGO_BUILD_JOBS':'4'}
    accuracy_build.run(['cargo', 'build', '--release', '--locked', '--manifest-path',
                        str(root/'runner/Cargo.toml')], root, 'timing-build', env)
    identity = json.loads((root/'build-identity.json').read_text())
    identity.update(executableSha256=accuracy_build.digest(Path(identity['executable'])),
                    timingRunnerSha256=accuracy_build.digest(HERE/'runner.rs'))
    from run import MODEL, MODEL_SHA
    assert accuracy_build.digest(MODEL) == MODEL_SHA
    tensors, geometry = metadata(MODEL)
    prefix = 'stt.parakeet.encoder.'
    assert geometry[prefix+'att_chunk_left_choices'] == [70]
    assert geometry[prefix+'att_chunk_chunk_choices'] == [1, 2, 7, 13]
    assert geometry[prefix+'att_chunk_right_choices'] == [0, 1, 2, 3, 4, 7, 13]
    assert (geometry[prefix+'subsampling_factor'] * geometry['stt.frontend.hop_length']
            * 1000 // geometry['stt.frontend.sample_rate']) == 80
    identity.update(modelSha256=MODEL_SHA, modelBytes=MODEL.stat().st_size,
                    tensors=tensors, geometryMetadata=geometry,
                    geometryMs={'default':[5600,1040,1040], 'low160':[5600,160,160]},
                    sourceHashes={str(p.relative_to(root)):accuracy_build.digest(p) for p in [
                        root/'runner/src/hex_pause.rs',
                        root/'runner/src/transcription/pause_segmenter.rs',
                        root/'transcribe/src/arch/parakeet/model.cpp',
                        root/'transcribe/bindings/rust/transcribe-cpp/src/family.rs']})
    (root/'timing-build-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print('timing identity and model geometry saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.work)
