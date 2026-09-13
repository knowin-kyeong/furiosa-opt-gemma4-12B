"""gen_tile_arms.py <projection.rs> <ops.rs> -- V380: attention O-weight tile-ratio arms t88, t104, t72.

Copies `project_output` (the V313 symmetric 96/24 path) with the constants 96 -> R and 24 -> 120 - R on code lines only
(word-bounded, comment lines untouched) and asserts that every copy substitutes exactly as many sites as the original
has, so a generator slip cannot silently leave a tile half-changed (V259's lesson). Adds one ops arm per ratio that
differs from `sliding_attention_output` only in the projection call.
"""
import io
import re
import sys

proj_path, ops_path = sys.argv[1], sys.argv[2]
ARMS = [88, 104, 72]


def subst(text, r):
    out, n96, n24 = [], 0, 0
    for line in text.split("\n"):
        if line.lstrip().startswith("//"):
            out.append(line)
            continue
        line, a = re.subn(r"\b96\b", str(r), line)
        line, b = re.subn(r"\b24\b", str(120 - r), line)
        n96 += a
        n24 += b
        out.append(line)
    return "\n".join(out), n96, n24


p = io.open(proj_path, encoding="utf-8").read()
start = p.index("pub(crate) fn project_output(")
end = p.index("\n}\n", start) + 3
fn = p[start:end]
same, n96, n24 = subst(fn, 96)
assert same == fn and n96 > 0 and n24 > 0, (n96, n24)

copies = []
for r in ARMS:
    assert r % 4 == 0 and (120 - r) % 4 == 0, r
    body, a, b = subst(fn, r)
    assert (a, b) == (n96, n24), (r, a, b, n96, n24)
    body = body.replace("pub(crate) fn project_output(", "pub(crate) fn project_output_t%d(" % r, 1)
    copies.append("/// V380: `project_output` with O-weight tiles %d/%d instead of 96/24 (constants only).\n%s" % (r, 120 - r, body))
p = p[:end] + "\n" + "\n".join(copies) + p[end:]
io.open(proj_path, "w", encoding="utf-8", newline="\n").write(p)

o = io.open(ops_path, encoding="utf-8").read()
attr = "#[device(chip = 1)]\n"
ks = o.index("pub fn sliding_attention_output(")
assert o[ks - len(attr):ks] == attr, "device attribute"
ke = o.index("\n}\n", ks) + 3
kfn = o[ks:ke]
call = "sliding::projection::project_output(ctx"
assert kfn.count(call) == 1, "projection call"
arms = []
for r in ARMS:
    body = kfn.replace("pub fn sliding_attention_output(", "pub fn sliding_attention_output_t%d(" % r, 1)
    body = body.replace(call, "sliding::projection::project_output_t%d(ctx" % r, 1)
    arms.append("/// V380 arm t%d: the attention output with O-weight tiles %d/%d.\n%s%s" % (r, r, 120 - r, attr, body))
o = o[:ke] + "\n" + "\n".join(arms) + o[ke:]
io.open(ops_path, "w", encoding="utf-8", newline="\n").write(o)
print("V380 tile arms", ARMS, "substitution sites per copy: 96 ->", n96, "24 ->", n24)
