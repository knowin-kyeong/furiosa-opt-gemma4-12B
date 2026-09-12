"""beampath.py <beam_trace> [grep] -- the beam-search path the scheduler kept: node names in decision order.

The trace is {step: [entry, ...]}; each entry has state_key, parent_state_key, selected_candidate ("T85 := Dma.DtoS(T84)"),
current_total_cycle, survived, is_finished_candidate. The kept schedule is the finished candidate with the smallest
current_total_cycle at the last step; walk parent_state_key back one step at a time."""
import json
import re
import sys

path = sys.argv[1]
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
d = json.load(open(path))
steps = sorted(int(k) for k in d)


def key(s):
    return (s['runnable_nodes_hash'], s['spilled_tensors_hash']) if s else None


by_step = {}
for st in steps:
    by_step[st] = {}
    for e in d[str(st)]:
        by_step[st].setdefault(key(e['state_key']), []).append(e)

last = steps[-1]
fin = [e for e in d[str(last)] if e.get('is_finished_candidate')] or [e for e in d[str(last)] if e.get('survived')] or d[str(last)]
best = min(fin, key=lambda e: e['current_total_cycle'])
chain = [best]
cur = best
for st in reversed(steps[:-1]):
    pk = key(cur['parent_state_key'])
    if pk is None:
        break
    cands = by_step[st].get(pk)
    if not cands:
        # the parent may sit more than one step back
        found = None
        for st2 in reversed([s for s in steps if s < st]):
            if pk in by_step[st2]:
                found = by_step[st2][pk]
                break
        if not found:
            break
        cands = found
    cur = min(cands, key=lambda e: e['current_total_cycle'])
    chain.append(cur)
chain.reverse()
print('steps', len(steps), 'path length', len(chain), 'final total cycle', best['current_total_cycle'])
for e in chain:
    c = e['selected_candidate']
    if not c:
        continue
    if pat and not pat.search(c):
        continue
    print('%5d %8d  %s' % (e['step'], e['current_total_cycle'], c[:160]))
