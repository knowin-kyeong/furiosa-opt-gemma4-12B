"""rates.py -- hardware DMA byte rate per command vs descriptor shape (V273 static schedules + rlir plots + V280 spans)."""
import json, re, sys
LOGS = ['span2_job.log', 'span3_job.log', 'span_bc.log', 'span_st.log', 'span_ts.log']
KER = {'sliding_project_qkv': 'V273Q.json', 'sliding_attention_output': 'V273A.json', 'decoder_feedforward': 'V273F.json'}


def src(i):
    m = re.search(r'--> (src/[^\s:]+:\d+)', i.get('description') or '')
    return m.group(1).replace('src/device/', '') if m else str(i['index'])


for k, sj in KER.items():
    d = json.load(open(sj, encoding='utf-8'))
    tens = {int(t['index']): t for t in d['tensors']}
    st = sorted([i for i in d['instructions'] if i['tpe'].startswith('Dma')], key=lambda i: (i['lifetime']['begin'], i['lifetime']['end'], str(i['index'])))
    rl = json.load(open('rlir_%s.json' % k, encoding='utf-8'))
    rins = rl['instructions'] if isinstance(rl, dict) and 'instructions' in rl else rl
    rd = {}
    for i in rins:
        if isinstance(i, dict) and str(i.get('tpe', '')).startswith('Dma'):
            rd[(i['lifetime']['begin'], i['lifetime']['end'])] = i.get('description') or ''
    hw = {}
    for lg in LOGS:
        rows = [l.rstrip('\n').split('\t') for l in open(lg, encoding='utf-8') if l.startswith('SPAN\t')]
        labs = sorted(set(r[1] for r in rows if r[1].startswith(k + '#')))
        for lab in labs:
            sp = sorted((int(r[2]), int(r[3])) for r in rows if r[1] == lab and r[5] == 'DMA')
            if len(sp) != len(st):
                continue
            for n, (b, e) in enumerate(sp):
                hw.setdefault(n, []).append(e - b)
    print('=====', k)
    for n, i in enumerate(st):
        b, e = i['lifetime']['begin'], i['lifetime']['end']
        if i['tpe'] == 'DmaStore':
            size = sum(tens[t]['size'] for t in i['input_tensors'] if t in tens and tens[t]['buffer_type'] == 'Sram')
        else:
            size = sum(tens[t]['size'] for t in i['output_tensors'] if t in tens)
        desc = rd.get((b, e), '')
        srcs = re.findall(r'^src\d+: (\[[^\]]*\])', desc, flags=re.M)
        dsts = re.findall(r'^dst\d+: (\[[^\]]*\])', desc, flags=re.M)
        ndesc = len(srcs)
        hws = sorted(hw.get(n, []))
        med = hws[len(hws) // 2] if hws else 0
        rate = size / max(med - 500, 1) if med else 0
        print('%-9s %-26s %9d B st %6d hw %6d (n=%d) %5.0f B/cy(-500) st %5.0f | nd=%d %s -> %s' % (
            i['tpe'][3:], src(i)[:26], size, e - b, med, len(hws), rate, size / max(e - b, 1), ndesc,
            (srcs[0] if srcs else desc.split('\n')[0])[:80], (dsts[0] if dsts else '')[:70]))
