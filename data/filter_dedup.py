"""Filter / remap / dedup pass for the v1 masked-encoder corpus (issue #160).

Settings (user-locked):
  - per-design: books already unpacked (Firebase via individual.txt; other
    sources unpacked here)
  - normalise 1.1 vanilla names -> 2.0 (target vocab is 2.0)
  - drop any design containing a non-vanilla entity (mods + Space Age),
    via the base-2.0 whitelist; combinators are vanilla so they stay
  - exact-layout dedup (byte-identical decoded layout, absolute coords)
  - window 50x50: report whole-fit vs crop, and total windows at stride 25
"""
import base64
import zlib
import json
import hashlib
import math
import time

WINDOW = 50
STRIDE = 25

# normalise 1.1 vanilla names -> 2.0 (target vocab is 2.0)
RENAME = json.load(open('data/normalise_map_1_1_to_2_0.json'))
# base-2.0 vanilla entity whitelist; anything outside it is a mod or Space Age
WHITELIST = {l.strip() for l in open('data/vanilla_whitelist.txt') if l.strip()}
COMBINATORS = {
    'arithmetic-combinator', 'decider-combinator', 'constant-combinator',
    'selector-combinator',
}

SOURCES = [  # (name, path, already_single_design)
    ('factorio.school',        'data/blueprints/factorio_school.individual.txt', True),
    ('factorio-blueprints.com','data/blueprints/factorio_blueprints_com.entries.txt', False),
    ('factoriocodex.com',      'data/blueprints/factoriocodex_com.entries.txt', False),
    ('fprints.xyz',            'data/blueprints/fprints_xyz.entries.txt', False),
]
OUT = 'data/blueprints/v1_clean.individual.txt'
STATS = 'data/blueprints/v1_clean.stats.json'


def read_strings(path):
    out, cur = [], None
    for line in open(path):
        line = line.rstrip('\n')
        if line[:1] == '0':
            if cur is not None:
                out.append(cur)
            cur = line
        elif cur is not None:
            cur += line
    if cur is not None:
        out.append(cur)
    return [s for s in out if s[:1] == '0']


def designs_of(s):
    """Yield individual blueprint dicts from a 0-string (book or single)."""
    try:
        bp = json.loads(zlib.decompress(base64.b64decode(s[1:])))
    except Exception:
        return
    stack = [bp]
    while stack:
        o = stack.pop()
        if 'blueprint_book' in o:
            stack.extend(o['blueprint_book'].get('blueprints', []))
        elif 'blueprint' in o and isinstance(o['blueprint'], dict):
            yield o['blueprint']


def encode(bp):
    return '0' + base64.b64encode(
        zlib.compress(json.dumps({'blueprint': bp}, separators=(',', ':')).encode(), 6)
    ).decode()


seen = set()
out_f = open(OUT, 'w')
stat = {'window': WINDOW, 'stride': STRIDE, 'per_source': {}, 'totals': {
    'designs_in': 0, 'undecodable': 0, 'empty': 0, 'dropped_nonvanilla': 0,
    'with_combinator': 0, 'renamed': 0, 'kept_pre_dedup': 0,
    'dup_collapsed': 0, 'final': 0, 'entities_final': 0,
    'fit_whole_50': 0, 'need_crop': 0, 'windows_50_stride25': 0,
}}
T = stat['totals']
t0 = time.time()

for name, path, single in SOURCES:
    ps = {'designs_in': 0, 'kept_new': 0, 'dup': 0, 'dropped': 0}
    strs = read_strings(path)
    for s in strs:
        for bp in designs_of(s):
            T['designs_in'] += 1; ps['designs_in'] += 1
            ents = bp.get('entities') or []
            if not ents:
                T['empty'] += 1; ps['dropped'] += 1; continue
            names = [e.get('name', '') for e in ents]
            renamed = False
            for e, nm in zip(ents, names):
                if nm in RENAME:
                    e['name'] = RENAME[nm]; renamed = True
            nameset = {e['name'] for e in ents}
            if not nameset <= WHITELIST:  # any mod / Space-Age entity -> drop design
                T['dropped_nonvanilla'] += 1; ps['dropped'] += 1; continue
            if nameset & COMBINATORS:
                T['with_combinator'] += 1
            if renamed:
                T['renamed'] += 1
            # exact-layout hash (absolute coords)
            key = hashlib.blake2b(json.dumps(
                sorted((e['name'], round(e['position']['x'], 3), round(e['position']['y'], 3),
                        e.get('direction', 0), e.get('recipe', '')) for e in ents),
                separators=(',', ':')).encode(), digest_size=16).digest()
            T['kept_pre_dedup'] += 1
            if key in seen:
                T['dup_collapsed'] += 1; ps['dup'] += 1; continue
            seen.add(key)
            ps['kept_new'] += 1; T['final'] += 1; T['entities_final'] += len(ents)
            xs = [e['position']['x'] for e in ents]; ys = [e['position']['y'] for e in ents]
            w = math.ceil(max(xs) - min(xs) + 1); h = math.ceil(max(ys) - min(ys) + 1)
            if max(w, h) <= WINDOW:
                T['fit_whole_50'] += 1; T['windows_50_stride25'] += 1
            else:
                T['need_crop'] += 1
                nx = max(1, math.ceil((w - WINDOW) / STRIDE) + 1)
                ny = max(1, math.ceil((h - WINDOW) / STRIDE) + 1)
                T['windows_50_stride25'] += nx * ny
            out_f.write(encode(bp) + '\n')
    stat['per_source'][name] = ps
    print(f"[{name}] in={ps['designs_in']} new={ps['kept_new']} dup={ps['dup']} "
          f"dropped={ps['dropped']}  ({time.time()-t0:.0f}s)", flush=True)

out_f.close()
json.dump(stat, open(STATS, 'w'), indent=2)
print("=== TOTALS ===", flush=True)
for k, v in T.items():
    print(f"  {k}: {v}", flush=True)
print(f"done in {time.time()-t0:.0f}s -> {OUT}", flush=True)
