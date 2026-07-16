"""Unpack Firebase blueprint records (books) into individual single-blueprint
0-prefixed strings, one per line. Re-encodes each sub-blueprint."""
import gzip
import json
import base64
import zlib
import time

SRC = 'data/raw/factorio_school.records.jsonl.gz'
OUT = 'data/blueprints/factorio_school.individual.txt'

t0 = time.time(); entries = 0; ind = 0; bad = 0
with gzip.open(SRC, 'rt') as fh, open(OUT, 'w') as out:
    for line in fh:
        r = json.loads(line)
        s = (r.get('blueprintString') or '').strip()
        if not s or s[0] != '0':
            continue
        entries += 1
        try:
            bp = json.loads(zlib.decompress(base64.b64decode(s[1:])))
        except Exception:
            bad += 1; continue
        stack = [bp]
        while stack:
            o = stack.pop()
            if 'blueprint_book' in o:
                stack.extend(o['blueprint_book'].get('blueprints', []))
            elif 'blueprint' in o and isinstance(o['blueprint'], dict):
                one = {'blueprint': o['blueprint']}
                enc = '0' + base64.b64encode(
                    zlib.compress(json.dumps(one, separators=(',', ':')).encode(), 6)
                ).decode()
                out.write(enc + '\n'); ind += 1
        if entries % 2000 == 0:
            print(f"{entries} entries -> {ind} individual  {time.time()-t0:.0f}s", flush=True)
print(f"DONE entries={entries} individual={ind} undecodable={bad} in {time.time()-t0:.0f}s", flush=True)
