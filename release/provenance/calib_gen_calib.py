"""Sample the target model's own greedy answers to the calibration prompts (run on gpu-8-vm against the live server).
Usage: gen_calib.py PROMPTS.json OUT.jsonl [MAX_TOKENS]"""
import json, sys, threading, queue, urllib.request, time
from transformers import AutoTokenizer
prompts = json.load(open(sys.argv[1])); out = sys.argv[2]; max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 400
model = open("/u701/j692wu/swift-local/model-path.txt").read().strip()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
q = queue.Queue(); lock = threading.Lock(); done = [0]; f = open(out, "w")
for i, p in enumerate(prompts): q.put((i, p))
def worker():
    while True:
        try: i, p = q.get_nowait()
        except queue.Empty: return
        ids = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        body = {"model": "swift", "prompt": ids, "max_tokens": max_tokens, "temperature": 0}
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r: d = json.loads(r.read().decode())
            text = d["choices"][0]["text"]; n = d["usage"]["completion_tokens"]
        except Exception as e:
            text, n = "", -1; sys.stderr.write("request %d failed: %r\n" % (i, e))
        with lock:
            f.write(json.dumps({"i": i, "lang": p["lang"], "domain": p["domain"], "prompt": p["text"], "output": text, "n_tokens": n}, ensure_ascii=False) + "\n"); f.flush()
            done[0] += 1
            if done[0] % 20 == 0: print("done", done[0], "/", len(prompts), flush=True)
t0 = time.time()
ts = [threading.Thread(target=worker) for _ in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
f.close(); print("finished", done[0], "prompts in %.0f s" % (time.time() - t0))
