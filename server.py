#!/usr/bin/env python3
"""
UBH MCL demo server. Python 3 standard library only, no installs.

  python3 server.py                  # serves on port 8080
  python3 server.py --key mysecret   # fixed admin key instead of a random one

Phones open  /            -> 3 flatlines until triggered
Trigger at   /admin?key=K -> START / RESET buttons
Or type  s + Enter (start) / r + Enter (reset) in this terminal.

Flow comes from the recorded MCL run (control.csv). Pump speed and pulmonary
pressure are MODELLED from that flow (see PRESSURE MODEL below).
Playback loops seamlessly; per-point noise is added on the phone.
"""
import argparse, csv, gzip, json, os, secrets, socket, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- config ----------------
FLOW_COL = "flow_msq_lpm"  # or "flow_sq_lpm" (noisier); columns of control.csv
OUT_FS = 50         # Hz sent to phones
PUMP_OFF_LPM = 1.0  # recording is cut before flow first drops below this (pump stop at ~55 s)
# PRESSURE MODEL (synthetic): P = gain * smoothed flow, with a small lag
RPM_MIN, RPM_MAX, RPM_LAG_S = 800.0, 1000.0, 0.0  # speed follows the flow waveform (same period/phase)
RPM_VIEW = (400.0, 1400.0)  # fixed y-axis for pump speed (wide, so the 800-1000 swing looks flat)
PUL_GAIN, PUL_LAG_S = 4.5, 0.35    # ~15 mmHg at 3.4 L/min
SMOOTH_S = 0.6
TARGET_BPM = 60.0  # playback is time-compressed so the flow wave repeats at this rate (None = real time)


# ---------------- data prep ----------------
def load_flow(path):
    """Reads control.csv: clean format (time_s,flow_msq_lpm,flow_sq_lpm) or a raw ControlDesk export."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip() == "time_s":
        hdr = [h.strip() for h in rows[0]]
        ti, qi = hdr.index("time_s"), hdr.index(FLOW_COL)
        data = [r for r in rows[1:] if len(r) > max(ti, qi) and r[ti].strip()]
        return [float(r[ti]) for r in data], [float(r[qi]) for r in data]
    # raw dSPACE export: trace rows are ",time,MSQ,SQ" after the "trace_values" marker
    col = 2 if FLOW_COL == "flow_msq_lpm" else 3
    i = next(k for k, r in enumerate(rows) if r and r[0].strip() == "trace_values")
    data = [r for r in rows[i:] if len(r) > col and r[1].strip() and r[col].strip()]
    return [float(r[1]) for r in data], [float(r[col]) for r in data]


def moving_avg(x, n):
    n = max(1, n)
    out, s = [], 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        out.append(s / min(i + 1, n))
    return out


def build_dataset(csv_path):
    t, q = load_flow(csv_path)
    dt = (t[-1] - t[0]) / (len(t) - 1)
    # cut before the pump stops (ignore the first 5 s)
    end = len(q)
    for i, v in enumerate(q):
        if t[i] > 5 and v < PUMP_OFF_LPM:
            end = max(0, i - int(1.0 / dt))
            break
    q = q[:end]
    # downsample
    step = max(1, round((1.0 / OUT_FS) / dt))
    q = q[::step]
    fs = 1.0 / (dt * step)

    # pick a loop end that matches the start (level + slope), in the last 10 s
    qs = moving_avg(q, int(0.3 * fs))
    lo = max(int(5 * fs), len(q) - int(10 * fs))
    s0 = qs[int(0.3 * fs)]
    d0 = qs[int(0.3 * fs) + 5] - s0
    best, best_i = 1e9, len(q) - 1
    for i in range(lo, len(q) - 6):
        cost = abs(qs[i] - s0) + 3 * abs((qs[i + 5] - qs[i]) - d0)
        if cost < best:
            best, best_i = cost, i
    q = q[:best_i]

    # crossfade the tail into the head so the wrap is seamless
    n = int(1.0 * fs)
    loop = q[n:]
    for j in range(n):
        w = (j + 1) / n
        loop[len(loop) - n + j] = (1 - w) * loop[len(loop) - n + j] + w * q[j]
    N = len(loop)

    # modelled pressures from circularly smoothed flow
    ext = loop[-int(2 * fs):] + loop
    sm = moving_avg(ext, int(SMOOTH_S * fs))[int(2 * fs):]
    def lagged(gain, lag_s):
        k = int(lag_s * fs)
        return [gain * sm[(i - k) % N] for i in range(N)]
    qlo, qhi = min(sm), max(sm)
    k_rpm = int(RPM_LAG_S * fs)
    rpm = [RPM_MIN + (sm[(i + k_rpm) % N] - qlo) / (qhi - qlo) * (RPM_MAX - RPM_MIN) for i in range(N)]
    pul_p = lagged(PUL_GAIN, PUL_LAG_S)

    # dominant period of the flow wave: first strong autocorrelation peak (avoids picking a multiple)
    mu = sum(sm) / N
    dz = [x - mu for x in sm]
    ac = lambda k: sum(dz[i] * dz[(i + k) % N] for i in range(0, N, 2))
    r0 = ac(0)
    rs = {k: ac(k) for k in range(int(1.5 * fs), int(10 * fs))}
    ks = sorted(rs)
    peaks = [k for k in ks[1:-1] if rs[k] >= rs[k - 1] and rs[k] >= rs[k + 1] and rs[k] > 0.3 * r0]
    best_k = peaks[0] if peaks else max(rs, key=rs.get)
    period_s = best_k / fs

    data = {
        "fs": fs,
        "seconds": N / fs,
        "period_s": period_s,
        "speed": (period_s * TARGET_BPM / 60.0) if TARGET_BPM else 1.0,
        "channels": [
            {"name": "Pump speed", "unit": "RPM", "v": [round(v) for v in rpm]},
            {"name": "Left flow", "unit": "L/min", "v": [round(v, 3) for v in loop]},
            {"name": "Pulmonary pressure", "unit": "mmHg", "v": [round(v, 2) for v in pul_p]},
        ],
    }
    for c in data["channels"]:
        if c["unit"] == "RPM":
            c["lo"], c["hi"] = RPM_VIEW
            continue
        lo, hi = min(c["v"]), max(c["v"])
        pad = 0.12 * (hi - lo)
        c["lo"], c["hi"] = round(lo - pad, 3), round(hi + pad, 3)
    return data


# ---------------- state ----------------
class State:
    def __init__(self):
        self.started = False
        self.start_ms = None
        self.version = 0
        self.viewers = 0
        self.cond = threading.Condition()

    def snapshot(self):
        now = time.time()
        viewers = self.viewers if SSE_ON else sum(1 for t in list(SEEN.values()) if now - t < 6)
        return {"started": self.started, "startMs": self.start_ms,
                "now": now * 1000, "viewers": viewers, "sse": SSE_ON}

    def start(self):
        with self.cond:
            if not self.started:
                self.started, self.start_ms = True, time.time() * 1000
                self.version += 1
                self.cond.notify_all()
        print("[trigger] START", flush=True)

    def reset(self):
        with self.cond:
            self.started, self.start_ms = False, None
            self.version += 1
            self.cond.notify_all()
        print("[trigger] RESET", flush=True)


STATE = State()
ADMIN_KEY = ""
SSE_ON = True
SEEN = {}  # poll-only viewer count: client id -> last poll time
FILES = {}  # path -> (content_type, raw_bytes, gzip_bytes)


def add_file(path, ctype, raw):
    FILES[path] = (ctype, raw, gzip.compress(raw, 9))


ADMIN_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCL admin</title>
</head>
<body>
<h2>MCL admin</h2>
<p>Status: <b id="st">...</b></p>
<p>Phones connected: <span id="v">0</span></p>
<button id="s">Start</button>
<button id="r">Reset</button>
<script>
const K = new URLSearchParams(location.search).get('key') || '';

async function post(p) {
  try {
    const r = await fetch(p + '?key=' + encodeURIComponent(K), {method: 'POST'});
    if (!r.ok) alert('failed: ' + r.status);
  } catch (e) {
    alert('network error');
  }
  poll();
}

async function poll() {
  try {
    const s = await (await fetch('/state', {cache: 'no-store'})).json();
    document.getElementById('st').textContent = s.started
      ? 'running (started ' + new Date(s.startMs).toLocaleTimeString() + ')'
      : 'not started';
    document.getElementById('v').textContent = s.viewers;
  } catch (e) {
    document.getElementById('st').textContent = 'cant reach server';
  }
}

document.getElementById('s').onclick = () => post('/start');
document.getElementById('r').onclick = () => { if (confirm('reset everyone back to flatline?')) post('/reset'); };
poll();
setInterval(poll, 2000);
</script>
</body>
</html>"""


# ---------------- http ----------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ubh"

    def log_message(self, *a):
        pass  # silence per-request logs (hundreds of phones)

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", gz=None, cache=False):
        use_gz = gz is not None and "gzip" in (self.headers.get("Accept-Encoding") or "")
        payload = gz if use_gz else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=300" if cache else "no-store")
        if use_gz:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(payload)

    def _authed(self):
        q = parse_qs(urlparse(self.path).query)
        return secrets.compare_digest(q.get("key", [""])[0], ADMIN_KEY)

    def do_GET(self):
        p = urlparse(self.path).path
        if p in FILES:
            ctype, raw, gz = FILES[p]
            return self._send(200, raw, ctype, gz, cache=(p == "/data.json"))
        if p == "/state":
            cid = parse_qs(urlparse(self.path).query).get("id", [""])[0][:16]
            if cid:
                SEEN[cid] = time.time()
            return self._send(200, json.dumps(STATE.snapshot()).encode(), "application/json")
        if p == "/admin":
            if not self._authed():
                return self._send(403, b"bad key")
            return self._send(200, ADMIN_HTML.encode(), "text/html; charset=utf-8")
        if p == "/events" and SSE_ON:
            return self._sse()
        self._send(404, b"not found")

    def do_POST(self):
        p = urlparse(self.path).path
        if p not in ("/start", "/reset"):
            return self._send(404, b"not found")
        if not self._authed():
            return self._send(403, b"bad key")
        STATE.start() if p == "/start" else STATE.reset()
        self._send(200, json.dumps(STATE.snapshot()).encode(), "application/json")

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        with STATE.cond:
            STATE.viewers += 1
        try:
            seen = -1
            while True:
                with STATE.cond:
                    if seen == STATE.version:
                        STATE.cond.wait(timeout=15)
                    changed = seen != STATE.version
                    seen = STATE.version
                    snap = STATE.snapshot()
                msg = ("data: " + json.dumps(snap) + "\n\n") if changed else ": hb\n\n"
                self.wfile.write(msg.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with STATE.cond:
                STATE.viewers -= 1


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def stdin_loop():
    for line in sys.stdin:
        c = line.strip().lower()
        if c == "s":
            STATE.start()
        elif c == "r":
            STATE.reset()


def main():
    global ADMIN_KEY, SSE_ON
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--csv", default=os.path.join(HERE, "control.csv"))
    ap.add_argument("--key", default=None, help="admin key (default: random)")
    ap.add_argument("--poll-only", action="store_true",
                    help="no SSE, phones poll every 2 s (use with Cloudflare quick tunnels)")
    a = ap.parse_args()

    ADMIN_KEY = a.key or secrets.token_urlsafe(6)
    SSE_ON = not a.poll_only
    data = build_dataset(a.csv)
    add_file("/data.json", "application/json", json.dumps(data, separators=(",", ":")).encode())
    with open(os.path.join(HERE, "index.html"), "rb") as f:
        add_file("/", "text/html; charset=utf-8", f.read())

    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.request_queue_size = 256
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    ip = lan_ip()
    print(f"Flow period {data['period_s']:.2f} s -> playback x{data['speed']:.2f} "
          f"({TARGET_BPM or 60 / data['period_s']:.0f} bpm)")
    print(f"Loop: {data['seconds']:.1f} s @ {data['fs']:.0f} Hz | data.json "
          f"{len(FILES['/data.json'][2]) / 1024:.1f} KB gzipped")
    print(f"Phones (same network): http://{ip}:{a.port}/")
    print(f"Trigger page:          http://{ip}:{a.port}/admin?key={ADMIN_KEY}")
    print("Mode: " + ("SSE + poll" if SSE_ON else "poll-only (2 s)"))
    print("Type s + Enter to START, r + Enter to RESET. Ctrl+C to quit.", flush=True)
    if sys.stdin and sys.stdin.isatty():
        threading.Thread(target=stdin_loop, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()