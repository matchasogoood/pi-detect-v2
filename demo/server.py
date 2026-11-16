"""server.py : local web server for the demo (standard library only).

  python demo/server.py                    # the real system (needs the datasets + a first-time FT-C fine-tune)
  python demo/server.py --stub             # fake numbers, to test the page only (red banner on the page)
Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = None


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html")
        if path == "/api/config":
            return self._send(200, ENGINE.config())
        if path == "/api/sample":
            from urllib.parse import parse_qs, urlparse
            kind = parse_qs(urlparse(self.path).query).get("kind", ["any"])[0]
            try:
                return self._send(200, ENGINE.sample(kind))
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        if path == "/api/scenario":
            from urllib.parse import parse_qs, urlparse
            kind = parse_qs(urlparse(self.path).query).get("kind", ["benign"])[0]
            try:
                return self._send(200, ENGINE.scenario(kind))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        if path == "/api/examples":
            return self._send(200, json.loads((HERE / "examples.json").read_text(encoding="utf-8")))
        if path == "/api/glossary":
            return self._send(200, json.loads((HERE / "glossary.json").read_text(encoding="utf-8")))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        try:
            if self.path == "/api/new":
                sid = uuid.uuid4().hex[:8]
                ENGINE.reset(sid)
                return self._send(200, {"session": sid})
            if self.path == "/api/reset":
                ENGINE.reset(data["session"])
                return self._send(200, {"ok": True})
            if self.path == "/api/step":
                text = str(data.get("text", ""))[:20000]
                ch = "indirect" if data.get("channel") == "indirect" else "direct"
                if not text.strip():
                    return self._send(400, {"error": "empty text"})
                return self._send(200, ENGINE.step(data["session"], text, ch, data.get("source") or None))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="fake numbers (page testing only)")
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--raw-dir", default=None, help="folder with the downloaded datasets (default: data_raw/)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    if a.stub:
        from stub_engine import StubEngine
        ENGINE = StubEngine()
    else:
        from engine import Engine
        ENGINE = Engine(seed=a.seed, raw_dir=a.raw_dir)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"demo ready -> http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
