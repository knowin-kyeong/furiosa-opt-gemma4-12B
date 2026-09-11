"""arm.py <test_fn> <tag> <ops_fn> | arm.py --sweep <CONST> <tag...> -- harness test-file edits.

Arm mode clones the `""` arm of `async fn <test_fn>` into a `"<tag>"` arm launching `ops::<ops_fn>`.
Sweep mode sets `<CONST>` (FFN_SWEEP / QKV_SWEEP / ATTN_SWEEP): with one tag "3" it is `[""; 3]`; with a base and one
test tag it is base,test,test,base x4; with more tags every tag appears 8 times in rotation. Use _ for "".
"""
import io
import re
import sys

p = 'tests/test_kernels.rs'
t = io.open(p, encoding='utf-8').read()
if sys.argv[1] == '--sweep':
    const, tags = sys.argv[2], [('' if x == '_' else x) for x in sys.argv[3:]]
    if tags == ['3']:
        body = 'const %s: &[&str] = &[""; 3];' % const
    else:
        if len(tags) == 2:
            order = [tags[0], tags[1], tags[1], tags[0]] * 4
        else:
            order = []
            for r in range(8):
                order += tags[r % len(tags):] + tags[:r % len(tags)]
        body = 'const %s: &[&str] = &[\n    %s,\n];' % (const, ', '.join('"%s"' % v for v in order))
    t, n = re.subn(r'const %s: &\[&str\] = (?:&\[""; 3\]|&\[\n(?:    [^\n]*\n)+\]);' % const, body, t)
    assert n == 1, const
    print('sweep', const, '=', body.replace('\n', ' '))
else:
    fn, tag, ops_fn = sys.argv[1:4]
    mark = '            other => panic!("no variant `{other}` for %s"),' % fn
    i = t.index(mark)
    start = t.index('            "" => {', t.index('async fn %s(' % fn))
    assert start < i
    end = t.index('\n            }\n', start) + len('\n            }\n')
    tmpl = t[start:end]
    assert tmpl.count('ops::%s,' % fn) == 1, 'template'
    assert ('            "%s" => {' % tag) not in t, 'tag exists'
    t = t[:i] + tmpl.replace('            "" => {', '            "%s" => {' % tag, 1).replace(
        'ops::%s,' % fn, 'ops::%s,' % ops_fn, 1) + t[i:]
    print('arm', tag, '->', ops_fn)
io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
