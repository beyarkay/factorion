"""Download factorio-blueprints.com blueprint strings via /api/blueprint/<id>/string.
IDs come from its sitemaps (/tmp/fbc_ids.txt). Writes entry-level 0-prefixed
strings + parallel metadata. Polite concurrency (8)."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

IDS = [l.strip() for l in open('/tmp/fbc_ids.txt') if l.strip()]
OUT = 'data/blueprints/factorio_blueprints_com.entries.txt'
META = 'data/blueprints/factorio_blueprints_com.entries.meta.jsonl'
URL = "https://factorio-blueprints.com/api/blueprint/{}/string"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

def fetch(bid):
    for _ in range(3):
        try:
            req = urllib.request.Request(URL.format(bid), headers={'User-Agent': UA})
            d = json.load(urllib.request.urlopen(req, timeout=30))
            s = d.get('blueprint') or d.get('blueprintString') or ''
            return bid, s.strip()
        except Exception:
            time.sleep(1)
    return bid, None

t0 = time.time(); ok = 0; fail = 0
with open(OUT, 'w') as out, open(META, 'w') as meta, ThreadPoolExecutor(max_workers=8) as ex:
    for i, (bid, s) in enumerate(ex.map(fetch, IDS), 1):
        if s and s[0] == '0':
            out.write(s + '\n'); meta.write(json.dumps({'id': bid}) + '\n'); ok += 1
        else:
            fail += 1
        if i % 1000 == 0:
            print(f"{i}/{len(IDS)} ok={ok} fail={fail} {time.time()-t0:.0f}s", flush=True)
print(f"DONE ok={ok} fail={fail} in {time.time()-t0:.0f}s", flush=True)
