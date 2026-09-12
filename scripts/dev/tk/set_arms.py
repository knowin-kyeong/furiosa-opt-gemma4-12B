"""set_arms.py <harness.rs> <shim fn> <SWEEP_CONST> <tag:ops_fn>... -- replace every named arm of one shim.

Removes all arms except "" from `async fn <shim fn>`, clones the "" arm once per <tag:ops_fn>, and sets <SWEEP_CONST> to
every tag plus production, 4 rows, each row rotated by one (the shim's own minute rotation shifts the start per job).
"""
import io
import re
import sys

path, shim_fn, const, arms = sys.argv[1], sys.argv[2], sys.argv[3], [a.split(':') for a in sys.argv[4:]]
t = io.open(path, encoding='utf-8').read()
shim = t.index("async fn %s(" % shim_fn)
shim_end = t.index("\n}\n", shim)
body = t[shim:shim_end]
tmpl_start = body.index('            "" => {')
tmpl_end = body.index('\n            }\n', tmpl_start) + len('\n            }\n')
tmpl = body[tmpl_start:tmpl_end]
assert tmpl.count('ops::%s,' % shim_fn) == 1, "template"
mark = '            other => panic!("no variant `{other}` for %s"),' % shim_fn
mark_i = body.index(mark)
clones = "".join(
    tmpl.replace('            "" => {', '            "%s" => {' % tag, 1).replace('ops::%s,' % shim_fn, 'ops::%s,' % fn, 1)
    for tag, fn in arms
)
body = body[:tmpl_end] + clones + body[mark_i:]
t = t[:shim] + body + t[shim_end:]

tags = [tag for tag, _ in arms] + [""]
rows = []
for r in range(4):
    row = tags[r % len(tags):] + tags[:r % len(tags)]
    rows.append("    " + ", ".join('"%s"' % x for x in row) + ",")
new = "const %s: &[&str] = &[\n%s\n];" % (const, "\n".join(rows))
t, n = re.subn(r'const %s: &\[&str\] = (?:&\[""; \d+\]|&\[\n(?:    [^\n]*\n)+\]);' % const, new, t)
assert n == 1, const
io.open(path, 'w', encoding='utf-8', newline='\n').write(t)
print("arms", tags, "->", path)
