"""Deterministic summaries from retained observations, independent of references."""
import re
import statistics


def tokens(text):
    return re.findall('[a-z0-9]+', text.lower())


def suffix_edit(old, new):
    prefix = 0
    for a, b in zip(old, new):
        if a != b:
            break
        prefix += 1
    return len(old)-prefix, len(new)-prefix


def event(snapshot):
    if snapshot is None:
        return None
    return {k:snapshot[k] for k in ['phase','wallMs','inputSamples','modelSamples']}


def summarize(row, snapshots, feeds):
    def first(field):
        return next((s for s in snapshots if s[field].strip()), None)

    useful = None
    final_tokens = tokens(row['recognizerText'])
    for i, s in enumerate(snapshots):
        if s['phase'] == 'finalize':
            break
        if len(final_tokens) >= 2 and all(tokens(x['full'])[:2] == final_tokens[:2]
                                         for x in snapshots[i:]):
            useful = s
            break
    revisions = rewrites = words_removed = words_added = chars_removed = chars_added = 0
    committed_violations = public_increments = 0
    previous_full = previous_committed = ''
    seen_text = False
    for s in snapshots:
        full = s['full']
        if full != previous_full:
            revisions += int(seen_text)
            removed, added = suffix_edit(tokens(previous_full), tokens(full))
            words_removed += removed
            words_added += added
            rewrites += int(removed > 0)
            removed, added = suffix_edit(previous_full, full)
            chars_removed += removed
            chars_added += added
        committed_violations += int(not s['committed'].startswith(previous_committed))
        previous_full, previous_committed = full, s['committed']
        seen_text |= bool(full.strip())
    for a, b in zip(snapshots, snapshots[1:]):
        public_increments += max(0, b['revision']-a['revision'])
    first_any = first('full')
    onset = row['energyOnsetProxyMs']
    return {
        'firstAny':event(first_any), 'firstTentative':event(first('tentative')),
        'firstCommitted':event(first('committed')), 'firstUseful':event(useful),
        'textBeforeEnergyProxy': bool(first_any and onset is not None and
                                      first_any['inputSamples']/16 < onset),
        'textRevisions':revisions, 'wordRewriteTransitions':rewrites,
        'wordsRemoved':words_removed, 'wordsAdded':words_added,
        'charsRemoved':chars_removed, 'charsAdded':chars_added,
        'committedPrefixViolations':committed_violations,
        'publicRevisionIncrements':public_increments,
        'endScheduledToFinalMs':row['finalWallMs']-row['scheduledEndMs'],
        'endReleaseToFinalMs':row['finalWallMs']-row['actualEndReleaseMs'],
        'flushServiceMs':row['finalWallMs']-row['flushStartMs'],
        'wouldMiss1500MsDeadline':row['finalWallMs']-row['actualEndReleaseMs'] > 1500,
        'wouldFallback':bool(row['error'] or not row['recognizerText'].strip() or
                            row['finalWallMs']-row['actualEndReleaseMs'] > 1500),
        'maxReleaseLatenessMs':max(f[1]-f[0]/16 for f in feeds),
        'maxQueueWaitMs':max(f[2]-f[1] for f in feeds),
        'maxFeedCallMs':max(f[3]-f[2] for f in feeds),
    }


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n':0, 'missing':None, 'min':None, 'median':None, 'max':None}
    return {'n':len(values), 'min':min(values), 'median':statistics.median(values),
            'max':max(values)}


def aggregate(rows):
    result = {'takes':len(rows), 'contended':sum(r['load']['contended'] for r in rows),
              'errors':sum(r['error'] is not None for r in rows),
              'empty':sum(not r['recognizerText'].strip() for r in rows)}
    for field in ['firstAny','firstTentative','firstCommitted','firstUseful']:
        events = [r['timing'][field] for r in rows if r['timing'][field]]
        result[field] = {'wallMs':distribution([e['wallMs'] for e in events]),
                         'inputMs':distribution([e['inputSamples']/16 for e in events]),
                         'missing':len(rows)-len(events),
                         'duringEndHandling':sum(e['phase'] != 'feed' for e in events)}
    for key in ['endScheduledToFinalMs','endReleaseToFinalMs','flushServiceMs',
                'textRevisions','wordRewriteTransitions','wordsRemoved','wordsAdded',
                'charsRemoved','charsAdded','maxReleaseLatenessMs','maxQueueWaitMs','maxFeedCallMs']:
        result[key] = distribution([r['timing'][key] for r in rows])
    for key in ['wouldMiss1500MsDeadline','wouldFallback','committedPrefixViolations',
                'textBeforeEnergyProxy']:
        result[key] = sum(r['timing'][key] for r in rows)
    return result
