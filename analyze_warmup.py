import csv

rows = []
with open("results/vanilla_sweep.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

def bler(r): return float(r['bler'])
def ber(r):  return float(r['ber'])

META = [
    ('warmup_sched_5x6',          '5c', '[5,5,5,5,5,5]       (ref)'),
    ('warmup_sched_8_4_4_4_4_6',  '5c', '[8,4,4,4,4,6]  mild warm-up'),
    ('warmup_sched_10_4_4_4_4_4', '5c', '[10,4,4,4,4,4] strong warm-up'),
    ('warmup_sched_10_5_5_5_5',   '4c', '[10,5,5,5,5]   warm-up+fewer'),
    ('warmup_sched_6x5',          '4c', '[6,6,6,6,6]    uniform'),
]
EBNOS = ['0.50', '0.60', '0.70']


def best_dn(tag, eb):
    cands = [r for r in rows if r['tag'] == tag and r['mode'] == 'denoiser'
             and r['ebno_db'] == eb and abs(float(r['beta'])) < 1e-9]
    return min(cands, key=bler) if cands else None


# TABLE 1
print("=" * 78)
print("TABLE 1  warm-up ablation: best BLER per schedule (beta=0, 320 blocks)")
print("  All schedules: total 30 BP iterations")
print("=" * 78)
print("  {:<40} {:>5} | {:>9} | {:>9} | {:>9}".format(
    "schedule", "calls", "Eb/N0=0.50", "Eb/N0=0.60", "Eb/N0=0.70"))
print("  " + "-" * 76)
best_overall = {eb: (999.0, '') for eb in EBNOS}
for tag, calls, label in META:
    vals = []
    for eb in EBNOS:
        d = best_dn(tag, eb)
        v = bler(d) if d else float('nan')
        vals.append(v)
        if v < best_overall[eb][0]:
            best_overall[eb] = (v, label)
    mark = " <-- ref" if tag == 'warmup_sched_5x6' else ""
    print("  {:<40} {:>5} | {:>9.4f} | {:>9.4f} | {:>9.4f}{}".format(
        label, calls, vals[0], vals[1], vals[2], mark))
print("  " + "-" * 76)
print("  {:<40} {:>5} | {:>9.4f} | {:>9.4f} | {:>9.4f}  <- overall best".format(
    "BEST per Eb/N0", "",
    best_overall['0.50'][0], best_overall['0.60'][0], best_overall['0.70'][0]))
for eb in EBNOS:
    print("    @{}: {}".format(eb, best_overall[eb][1]))

# TABLE 2
print()
print("=" * 78)
print("TABLE 2  detail @ Eb/N0=0.60  (beta=0, all alpha-schedules)")
print("=" * 78)
for tag, calls, label in META:
    cands = [r for r in rows if r['tag'] == tag and r['mode'] == 'denoiser'
             and r['ebno_db'] == '0.60' and abs(float(r['beta'])) < 1e-9]
    if not cands:
        continue
    cands.sort(key=bler)
    print("  {} ({})".format(label, calls))
    for i, r in enumerate(cands):
        flag = " <- best" if i == 0 else ""
        print("    alpha={:<32}  BLER={:.4f}  BER={:.2e}{}".format(
            r['alpha_schedule'], bler(r), ber(r), flag))

# TABLE 3
print()
print("=" * 78)
print("TABLE 3  BLER ratio vs [5x6] reference  (beta=0 best, < 1.0 = BETTER)")
print("=" * 78)
print("  {:<40} {:>5} | {:>10} | {:>10} | {:>10} | verdict".format(
    "schedule", "calls", "@0.50", "@0.60", "@0.70"))
print("  " + "-" * 80)
ref56 = {eb: best_dn('warmup_sched_5x6', eb) for eb in EBNOS}
for tag, calls, label in META:
    if tag == 'warmup_sched_5x6':
        continue
    ratios = []
    for eb in EBNOS:
        d = best_dn(tag, eb)
        r5 = ref56[eb]
        ratio = bler(d) / bler(r5) if (d and r5 and bler(r5) > 0) else float('nan')
        ratios.append(ratio)
    verdict = ("BETTER" if all(x < 1.0 for x in ratios if x == x) else
               "MIXED"  if any(x < 1.0 for x in ratios if x == x) else "WORSE")
    print("  {:<40} {:>5} | {:>10.3f} | {:>10.3f} | {:>10.3f} | {}".format(
        label, calls, ratios[0], ratios[1], ratios[2], verdict))
