import re, sys
path = sys.argv[1]
only_near_sync = len(sys.argv) > 2
src = open(path).read().split("\n")
tasks = [(i, l) for i, l in enumerate(src) if re.match(r"static void task_\d+_chip\d+_cluster\d+_\d+\(", l)]
print("tasks:", [l.split("(")[0].replace("static void ", "") for i, l in tasks])
ends = [i for i, _ in tasks[1:]] + [len(src)]
for (start, name), end in zip(tasks, ends):
    print("=" * 20, name.split("(")[0].replace("static void ", ""))
    blocks = []
    cur = None
    for i in range(start, end):
        l = src[i]
        m = re.match(r"\s*// (\d+) \(queue block, queue size (\d+)\)", l)
        if m:
            cur = [int(m.group(1)), int(m.group(2)), []]
            blocks.append(cur)
            continue
        if cur is None:
            continue
        s = l.strip()
        m = re.match(r"tail = (function_chip\d+_cluster\d+_\d+)\(tail, (.*)", s)
        if m:
            args = m.group(2)
            dmas = sorted(set(re.findall(r"\bdma_\d+\b", args)))
            addrs = sorted(set(re.findall(r"addrs\[[^\]]+\]", args)))
            tus = sorted(set(re.findall(r"\b(?:main|sub)_tu_\d+\b", args)))
            cur[2].append("%s%s%s%s" % (m.group(1).replace("function_chip0_", "f_"), (" dma=" + ",".join(dmas)) if dmas else "", (" arg=" + ",".join(a.replace("addrs", "") for a in addrs)) if addrs else "", (" tu=" + ",".join(tus)) if tus else ""))
            continue
        for pat in (r"(wait_dma)\(tail, (\d+)\)", r"(sync_intra_chip_cluster)\((\d+) \+", r"(wait_sync_intra_chip_cluster)\((\d+) \+", r"(tuc_reserve_resources)\(([^)]*)\)"):
            m = re.search(pat, s)
            if m:
                cur[2].append("%s(%s)" % (m.group(1), m.group(2)))
    for b in blocks:
        txt = " ; ".join(b[2])
        print("  blk %3d q%-3d %s" % (b[0], b[1], txt[:400]))
