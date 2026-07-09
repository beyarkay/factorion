import json
import gzip
import urllib.request
import time
import sys
from concurrent.futures import ThreadPoolExecutor

KEYS = list(json.load(open('/tmp/bp_summaries.json')).keys())
OUT = 'data/factorio_school_blueprints.jsonl.gz'
BASE = "https://facorio-blueprints.firebaseio.com/blueprints/{}.json"

def fetch(k):
    for _ in range(3):
        try:
            raw = urllib.request.urlopen(BASE.format(k), timeout=30).read()
            return k, raw
        except Exception:
            time.sleep(1)
    return k, None

t0=time.time(); ok=0; fail=0; nbytes=0
with gzip.open(OUT,'wb') as out, ThreadPoolExecutor(max_workers=16) as ex:
    for i,(k,raw) in enumerate(ex.map(fetch, KEYS),1):
        if raw is None:
            fail+=1; continue
        try:
            rec=json.loads(raw); rec['_key']=k
            line=(json.dumps(rec,separators=(',',':'))+'\n').encode()
            out.write(line); nbytes+=len(line); ok+=1
        except Exception:
            fail+=1
        if i%2000==0:
            print(f"{i}/{len(KEYS)} ok={ok} fail={fail} raw={nbytes/1e6:.0f}MB {time.time()-t0:.0f}s",flush=True)
print(f"DONE ok={ok} fail={fail} uncompressed={nbytes/1e6:.0f}MB in {time.time()-t0:.0f}s",flush=True)
