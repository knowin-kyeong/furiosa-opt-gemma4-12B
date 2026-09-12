"""armspans.py <log_prefix> <arm...> (stdin to the pod's python3) -- per-arm cluster-0 load span and end-sync (cluster-1 lag)
medians over /root/tk/<prefix>_r*.log (+ <prefix>_r0.log), for load-only probe arms."""
import glob
import statistics as st
import sys

prefix, arms = sys.argv[1], sys.argv[2:]
files = sorted(glob.glob('/root/tk/%s_r*.log' % prefix))
per = {a: {'win': [], 'load': [], 'sync': []} for a in arms}
for f in files:
    spans = {}
    for l in open(f, errors='replace'):
        if l.startswith('SPAN\t'):
            r = l.rstrip('\n').split('\t')
            spans.setdefault(r[1], []).append((int(r[2]), int(r[3]), r[5] if len(r) > 5 else r[-1]))
    for lab, sp in spans.items():
        if ' ' not in lab:
            continue
        arm = lab.split(' ')[1].split('#')[0]
        if arm not in per:
            continue
        t0 = min(s[0] for s in sp)
        t1 = max(s[1] for s in sp)
        dmas = [e - b for b, e, n in sp if n == 'DMA']
        syncs = [e - b for b, e, n in sp if n == 'Cluster']
        per[arm]['win'].append(t1 - t0)
        per[arm]['load'].append(max(dmas) if dmas else 0)
        per[arm]['sync'].append(syncs[-1] if syncs else 0)
print('files', len(files))
for a in arms:
    d = per[a]
    if d['win']:
        print('%-5s n=%-3d window med %6d | c0 load span med %6d | last sync (c1 lag) med %6d  (min %d, max %d)' % (
            a, len(d['win']), st.median(d['win']), st.median(d['load']), st.median(d['sync']), min(d['sync']), max(d['sync'])))
