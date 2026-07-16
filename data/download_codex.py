import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
IDS=[l.strip() for l in open('/tmp/fcx_ids.txt') if l.strip()]
URL="https://factoriocodex.com/api/v1/blueprints/{}"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
PAT=re.compile(r'0e[A-Za-z0-9+/=]{30,}')
def fetch(i):
    for _ in range(3):
        try:
            req=urllib.request.Request(URL.format(i),headers={'User-Agent':UA})
            s=urllib.request.urlopen(req,timeout=30).read().decode()
            m=PAT.search(s); return i,(m.group(0) if m else None)
        except Exception: time.sleep(1)
    return i,None
ok=0
with open('data/blueprints/factoriocodex_com.entries.txt','w') as out, ThreadPoolExecutor(max_workers=8) as ex:
    for i,s in ex.map(fetch,IDS):
        if s: out.write(s+'\n'); ok+=1
print(f"codex DONE ok={ok}/{len(IDS)}",flush=True)
