"""mk_base273.py <harness_tests.rs> -- on a V273_submit tree: install the sweep harness test file with every variant
arm except `""` removed and all three sweeps back to 3 launches, so `""` is exactly the shipped code and variant
kernels can be added next to it without copying any helper."""
import io
import re
import sys

t = io.open(sys.argv[1], encoding='utf-8').read()
ARM = re.compile(r'^            "([A-Za-z0-9_]*)" => \{\n', re.M)
out, pos, removed = [], 0, []
for m in ARM.finditer(t):
    if m.start() < pos:
        continue
    end = t.index('\n            }\n', m.end()) + len('\n            }\n')
    if m.group(1) == '':
        continue
    out.append(t[pos:m.start()])
    removed.append(m.group(1))
    pos = end
out.append(t[pos:])
t = ''.join(out)
for name in ('FFN_SWEEP', 'QKV_SWEEP', 'ATTN_SWEEP'):
    t, n = re.subn(r'const %s: &\[&str\] = (?:&\[""; 3\]|&\[\n(?:    [^\n]*\n)+\]);' % name,
                   'const %s: &[&str] = &[""; 3];' % name, t)
    assert n == 1, name
arms = ARM.findall(t)
assert arms == ['', '', ''], arms
io.open('tests/test_kernels.rs', 'w', encoding='utf-8', newline='\n').write(t)
print('harness base: removed arms', ' '.join(removed), '| kept 3 default arms, sweeps x3')
