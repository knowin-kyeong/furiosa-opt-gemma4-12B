"""map.py <span.log> <label-substring> <launch#> <schedule.json> -- align HW spans to static instructions by class order.

Classes: DMA (Dma* static / 'DMA' span), SYNC (ExplicitSync static / 'Cluster' span), TU (Main/Sub non-Noop static /
Renegade::* span). Within a class both lists are sorted by begin; the k-th static instruction is paired with the k-th span.
Prints the merged hardware timeline with static begin/dur, contexts and source line.
"""
import json
import re
import sys

log, labsub, launch, sched = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [l.rstrip('\n').split('\t') for l in open(log, encoding='utf-8') if l.startswith('SPAN\t')]
labs = []
for r in rows:
    if labsub in r[1] and r[1].endswith('#' + launch) and r[1] not in labs:
        labs.append(r[1])
lab = labs[0]
sp = sorted((int(r[2]), int(r[3]), r[5]) for r in rows if r[1] == lab)


def hcls(n):
    if n == 'DMA':
        return 'DMA'
    if n == 'Cluster':
        return 'SYNC'
    if n == 'Task':
        return None
    return 'TU'


def src(i):
    m = re.search(r'--> (src/[^\s:]+:\d+)', i.get('description') or '')
    return m.group(1).replace('src/device/', '') if m else str(i['index'])


def scls(i):
    if str(i['index']).startswith('ExplicitSync'):
        return 'SYNC'
    if i['tpe'].startswith('Dma'):
        return 'DMA'
    if i['tpe'] in ('Main', 'Sub') and 'Noop' not in (i.get('contexts') or []):
        return 'TU'
    return None


d = json.load(open(sched, encoding='utf-8'))
ins = sorted(d['instructions'], key=lambda i: (i['lifetime']['begin'], i['lifetime']['end'], str(i['index'])))
st = {c: [i for i in ins if scls(i) == c] for c in ('DMA', 'SYNC', 'TU')}
hw = {c: [s for s in sp if hcls(s[2]) == c] for c in ('DMA', 'SYNC', 'TU')}
print(lab, {c: (len(st[c]), len(hw[c])) for c in st})
tens = {int(t['index']): t for t in d.get('tensors', [])}
merged = []
for c in st:
    for k, s in enumerate(hw[c]):
        i = st[c][k] if k < len(st[c]) else None
        merged.append((s[0], s[1], s[2], i))
for b, e, n, i in sorted(merged, key=lambda x: (x[0], x[1])):
    if i is None:
        print('%7d-%-7d %6d %-14s ?' % (b, e, e - b, n[:14]))
        continue
    ctx = ','.join(x.replace('Context', '').replace('Engine', '') for x in (i.get('contexts') or []))
    sb, se = i['lifetime']['begin'], i['lifetime']['end']
    outs = ','.join('%s:%s' % (tens[t]['buffer_type'][:3], tens[t]['name'][:18]) for t in (i.get('output_tensors') or []) if t in tens)
    print('%7d-%-7d %6d %-14s | st %6d+%-5d %-5.1fx %-12s %-9s %-32s %s' % (
        b, e, e - b, n.replace('Renegade::', '')[:14], sb, se - sb, (e - b) / max(se - sb, 1), i['tpe'][:12], ctx[:9], src(i)[:32], outs[:60]))
