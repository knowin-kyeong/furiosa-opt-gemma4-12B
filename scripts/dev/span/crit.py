"""crit.py <span.log> <label-substring> <launch#> <schedule.json> [--all] -- hardware critical path of one launch.

Spans are paired with static instructions by class order (DMA / SYNC / TU, as map.py). For every instruction the binding
constraint is the latest of: each input tensor's hardware availability (writer end, through zero-duration alias nodes;
a tensor written by a DmaStore is available only after the ExplicitSync that follows the store), and the end of the
previous instruction on each queue it occupies (DMA FIFO, MainContext, SubContext, VectorEngine). The chain is traced
back from the last-ending instruction; `slack` = hardware start - binding time (large slack = waiting on something the
model does not see, e.g. a PE walking its list)."""
import json
import re
import sys

log, labsub, launch, sched = sys.argv[1:5]
ALL = '--all' in sys.argv
rows = [l.rstrip('\n').split('\t') for l in open(log, encoding='utf-8', errors='replace') if l.startswith('SPAN\t')]
labs = [r[1] for r in rows if labsub in r[1] and r[1].endswith('#' + launch)]
lab = labs[0]
sp = sorted((int(r[2]), int(r[3]), r[5]) for r in rows if r[1] == lab)


def hcls(n):
    return {'DMA': 'DMA', 'Cluster': 'SYNC', 'Task': None}.get(n, 'TU')


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
hw = {}
for c in ('DMA', 'SYNC', 'TU'):
    st = [i for i in ins if scls(i) == c]
    hs = [s for s in sp if hcls(s[2]) == c]
    for k, i in enumerate(st):
        if k < len(hs):
            hw[id(i)] = hs[k]
alias = {}
for i in ins:
    if scls(i) is None:
        for t in i.get('output_tensors') or []:
            alias.setdefault(t, set()).update(i.get('input_tensors') or [])
writer = {}
for i in ins:
    if scls(i) in ('DMA', 'TU'):
        for t in i.get('output_tensors') or []:
            writer.setdefault(t, []).append(i)
# the sync that follows each store (static order)
sync_after = {}
pending = []
for i in ins:
    if scls(i) == 'DMA' and i['tpe'] == 'DmaStore':
        pending.append(i)
    elif scls(i) == 'SYNC':
        for s in pending:
            sync_after[id(s)] = i
        pending = []


def queues(i):
    t, ctx = i['tpe'], i.get('contexts') or []
    q = []
    if t.startswith('Dma'):
        q.append('DMA')
    for c, n in (('MainContext', 'Main'), ('SubContext', 'Sub'), ('VectorEngine', 'VE')):
        if c in ctx:
            q.append(n)
    return q


def writers_of(t, seen):
    if t in seen:
        return []
    seen.add(t)
    out = list(writer.get(t, []))
    for s in alias.get(t, ()):
        out += writers_of(s, seen)
    return out


info = {}
last_on = {}
for i in ins:
    c = scls(i)
    if c not in ('DMA', 'TU') or id(i) not in hw:
        continue
    b, e, n = hw[id(i)]
    cands = []
    for t in i.get('input_tensors') or []:
        for w in writers_of(t, set()):
            if w is i or id(w) not in hw:
                continue
            if (w['lifetime']['begin'], w['lifetime']['end']) > (i['lifetime']['begin'], i['lifetime']['end']):
                continue
            we = hw[id(w)][1]
            cands.append((we, 'in:' + src(w), w))
            if w['tpe'] == 'DmaStore' and id(w) in sync_after and id(sync_after[id(w)]) in hw:
                sy = sync_after[id(w)]
                cands.append((hw[id(sy)][1], 'sync-after:' + src(w), sy))
    for q in queues(i):
        if q in last_on:
            p = last_on[q]
            cands.append((hw[id(p)][1], 'q%s:%s' % (q, src(p)), p))
    for q in queues(i):
        last_on[q] = i
    bind = max(cands, key=lambda x: x[0]) if cands else (0, 'start', None)
    info[id(i)] = (b, e, bind)

end_i = max((i for i in ins if id(i) in info), key=lambda i: info[id(i)][1])
print(lab, 'end', info[id(end_i)][1])
chain = []
cur = end_i
seen = set()
while cur is not None and id(cur) not in seen:
    seen.add(id(cur))
    if id(cur) in info:
        b, e, bind = info[id(cur)]
        chain.append((b, e, cur, bind))
        cur = bind[2]
    elif id(cur) in hw:  # a sync
        b, e, n = hw[id(cur)]
        chain.append((b, e, cur, (None, 'sync', None)))
        # the sync waits for its store: find the store it follows
        prev = [s for s in ins if id(s) in sync_after and sync_after[id(s)] is cur and id(s) in info]
        cur = prev[-1] if prev else None
    else:
        break
tot = {}
for b, e, i, bind in reversed(chain):
    slack = b - bind[0] if bind[0] is not None else 0
    k = scls(i) or '?'
    tot[k] = tot.get(k, 0) + (e - b)
    tot['slack'] = tot.get('slack', 0) + max(slack, 0)
    print('  %7d-%-7d %6d  %-5s %-10s %-30s  bound by %-40s slack %6d' % (b, e, e - b, k, i['tpe'][:10], src(i)[:30], bind[1][:40], slack))
print('  chain totals:', tot)
if ALL:
    for i in ins:
        if id(i) in info:
            b, e, bind = info[id(i)]
            print('   all %7d-%-7d %-10s %-30s bound %-40s slack %6d' % (b, e, i['tpe'][:10], src(i)[:30], bind[1][:40], b - bind[0]))
