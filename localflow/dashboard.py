"""Local web dashboard - the Wispr Flow-style app window.

Stdlib-only (http.server) so it runs anywhere, bound to localhost by default.
Serves the single-page UI from localflow/static/index.html: home with streak /
words / WPM / time-saved cards and recent activity, searchable history, the
personal dictionary, and live-saving settings.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .app import FlowController

_STATIC_DIR = Path(__file__).parent / "static"


def _load_page() -> str:
    index = _STATIC_DIR / "index.html"
    try:
        return index.read_text(encoding="utf-8")
    except OSError:
        return _PAGE  # fallback: minimal built-in page

_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LocalFlow</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#0f1115; --card:#181b22; --text:#e8eaf0; --muted:#8a91a3; --accent:#7aa2ff; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--text); font:15px/1.5 system-ui, sans-serif; padding:2rem; max-width:960px; margin:0 auto; }
h1 { font-size:1.4rem; margin-bottom:1rem; } h1 span { color:var(--accent); }
h2 { font-size:1.05rem; margin:1.6rem 0 .6rem; color:var(--muted); font-weight:600; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.8rem; }
.card { background:var(--card); border-radius:10px; padding:.9rem 1rem; }
.card b { display:block; font-size:1.5rem; } .card small { color:var(--muted); }
input, button { background:#232733; color:var(--text); border:1px solid #313747; border-radius:8px; padding:.45rem .7rem; font:inherit; }
button { cursor:pointer; } button:hover { border-color:var(--accent); }
table { width:100%; border-collapse:collapse; }
td { padding:.5rem .4rem; border-top:1px solid #262b38; vertical-align:top; }
td.time { color:var(--muted); white-space:nowrap; font-size:.85rem; }
td.app { color:var(--accent); font-size:.85rem; white-space:nowrap; }
.pill { display:inline-block; background:#232733; border-radius:999px; padding:.15rem .7rem; margin:.15rem .2rem .15rem 0; }
.pill button { border:none; padding:0 .2rem; color:var(--muted); background:none; }
.row { display:flex; gap:.5rem; margin:.4rem 0; flex-wrap:wrap; }
</style></head><body>
<h1>Local<span>Flow</span> dashboard</h1>
<div class="cards" id="stats"></div>
<h2>Personal dictionary</h2>
<div id="dict"></div>
<div class="row"><input id="newWord" placeholder="Add word or name"><button onclick="addWord()">Add</button></div>
<h2>Text replacements</h2>
<div id="repl"></div>
<div class="row"><input id="rFrom" placeholder="spoken"><input id="rTo" placeholder="written"><button onclick="addRepl()">Add</button></div>
<h2>History</h2>
<div class="row"><input id="q" placeholder="Search dictations" oninput="loadHistory()"><button onclick="clearHistory()">Clear all</button></div>
<table id="hist"></table>
<script>
async function j(url, opts) { const r = await fetch(url, opts); return r.json(); }
function esc(s) { const d = document.createElement('div'); d.innerText = s; return d.innerHTML; }
async function loadStats() {
  const s = await j('/api/stats');
  document.getElementById('stats').innerHTML = [
    ['Words dictated', s.total_words], ['Dictations', s.total_entries],
    ['Average WPM', s.average_wpm], ['Streak (days)', s.streak_days],
    ['Words today', s.words_today],
  ].map(([k,v]) => `<div class="card"><b>${v}</b><small>${k}</small></div>`).join('');
}
async function loadDict() {
  const d = await j('/api/dictionary');
  document.getElementById('dict').innerHTML = d.words.map(w =>
    `<span class="pill">${esc(w)}<button onclick="delWord('${esc(w)}')">✕</button></span>`).join('') || '<small>empty</small>';
  document.getElementById('repl').innerHTML = Object.entries(d.replacements).map(([k,v]) =>
    `<span class="pill">${esc(k)} → ${esc(v)}<button onclick="delRepl('${esc(k)}')">✕</button></span>`).join('') || '<small>none</small>';
}
async function loadHistory() {
  const q = document.getElementById('q').value;
  const h = await j('/api/history?q=' + encodeURIComponent(q));
  document.getElementById('hist').innerHTML = h.entries.map(e => `<tr>
    <td class="time">${new Date(e.timestamp*1000).toLocaleString()}</td>
    <td>${esc(e.formatted_text)}</td><td class="app">${esc(e.app||'')}</td>
    <td><button onclick="delEntry(${e.id})">✕</button></td></tr>`).join('');
}
async function addWord() {
  const w = document.getElementById('newWord').value.trim(); if (!w) return;
  await j('/api/dictionary', {method:'POST', body: JSON.stringify({add: w})});
  document.getElementById('newWord').value=''; loadDict();
}
async function delWord(w) { await j('/api/dictionary', {method:'POST', body: JSON.stringify({remove: w})}); loadDict(); }
async function addRepl() {
  const f = document.getElementById('rFrom').value.trim(), t = document.getElementById('rTo').value;
  if (!f || !t) return;
  await j('/api/dictionary', {method:'POST', body: JSON.stringify({replace_from: f, replace_to: t})});
  document.getElementById('rFrom').value=''; document.getElementById('rTo').value=''; loadDict();
}
async function delRepl(k) { await j('/api/dictionary', {method:'POST', body: JSON.stringify({remove_replacement: k})}); loadDict(); }
async function delEntry(id) { await j('/api/history/delete', {method:'POST', body: JSON.stringify({id})}); loadHistory(); loadStats(); }
async function clearHistory() { await j('/api/history/delete', {method:'POST', body: JSON.stringify({all: true})}); loadHistory(); loadStats(); }
loadStats(); loadDict(); loadHistory(); setInterval(loadStats, 10000); setInterval(loadHistory, 10000);
</script></body></html>
"""


class DashboardServer:
    def __init__(self, controller: FlowController, host: str = "127.0.0.1",
                 port: int = 5170, hotkey_recorder=None,
                 on_hotkeys_changed=None) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        # Optional daemon hooks: hotkey_recorder() blocks while capturing a
        # key combo and returns it as a string (or None on timeout);
        # on_hotkeys_changed() applies edited hotkeys to the live listener.
        self.hotkey_recorder = hotkey_recorder
        self.on_hotkeys_changed = on_hotkeys_changed
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------------------- server

    def start(self) -> int:
        controller = self.controller
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # quiet
                pass

            def _send(self, payload, status: int = 200, ctype: str = "application/json") -> None:
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                if url.path in ("/", "/index.html"):
                    self._send(_load_page().encode(), ctype="text/html; charset=utf-8")
                elif url.path == "/api/settings":
                    self._send(controller.config.to_dict())
                elif url.path == "/api/stats":
                    self._send(dataclasses.asdict(controller.history.stats()))
                elif url.path == "/api/history":
                    params = parse_qs(url.query)
                    q = (params.get("q") or [""])[0]
                    limit = int((params.get("limit") or ["100"])[0])
                    entries = (
                        controller.history.search(q, limit) if q
                        else controller.history.recent(limit)
                    )
                    self._send({"entries": [dataclasses.asdict(e) for e in entries]})
                elif url.path == "/api/dictionary":
                    self._send({
                        "words": controller.dictionary.words,
                        "replacements": controller.dictionary.replacements,
                    })
                elif url.path == "/api/llm":
                    params = parse_qs(url.query)
                    force = (params.get("refresh") or ["0"])[0] == "1"
                    available = controller.llm.probe(force=force)
                    self._send({
                        "available": available,
                        "base_url": controller.llm.base_url,
                        "model": controller.llm.model,
                        "models": controller.llm.models,
                    })
                elif url.path == "/api/state":
                    self._send({
                        "status": controller.state.status,
                        "hands_free": controller.state.hands_free,
                    })
                else:
                    self._send({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._send({"error": "bad json"}, status=400)
                    return
                if url.path == "/api/dictionary":
                    if data.get("add"):
                        controller.dictionary.add(str(data["add"]))
                    if data.get("remove"):
                        controller.dictionary.remove(str(data["remove"]))
                    if data.get("replace_from") and data.get("replace_to") is not None:
                        controller.dictionary.add_replacement(
                            str(data["replace_from"]), str(data["replace_to"])
                        )
                    if data.get("remove_replacement"):
                        controller.dictionary.replacements.pop(
                            str(data["remove_replacement"]), None
                        )
                    # persist into config
                    controller.config.dictionary = controller.dictionary.words
                    controller.config.replacements = controller.dictionary.replacements
                    controller.config.save()
                    self._send({"ok": True})
                elif url.path == "/api/settings":
                    self._apply_settings_patch(data)
                    controller.config.save()
                    if "hotkeys" in data and server.on_hotkeys_changed:
                        try:
                            server.on_hotkeys_changed()
                        except Exception:
                            pass  # next daemon restart picks them up anyway
                    self._send(controller.config.to_dict())
                elif url.path == "/api/hotkeys/record":
                    combo = None
                    if server.hotkey_recorder is not None:
                        try:
                            combo = server.hotkey_recorder()
                        except Exception:
                            combo = None
                    self._send({"combo": combo})
                elif url.path == "/api/history/delete":
                    if data.get("all"):
                        controller.history.clear()
                    elif data.get("id") is not None:
                        controller.history.delete(int(data["id"]))
                    self._send({"ok": True})
                else:
                    self._send({"error": "not found"}, status=404)

            def _apply_settings_patch(self, patch: dict) -> None:
                """Apply a partial config update: {"formatting": {"x": true}, ...}.

                Only keys that already exist on the config dataclasses are
                accepted; value types must match the current value's type.
                """
                cfg = controller.config
                for key, value in patch.items():
                    if key == "user_name" and isinstance(value, str):
                        cfg.user_name = value
                    elif isinstance(value, dict) and hasattr(cfg, key):
                        section = getattr(cfg, key)
                        if not dataclasses.is_dataclass(section):
                            continue
                        for sub_key, sub_value in value.items():
                            if not hasattr(section, sub_key):
                                continue
                            current = getattr(section, sub_key)
                            if current is None or isinstance(sub_value, type(current)) \
                                    or (isinstance(current, float) and isinstance(sub_value, (int, float))):
                                setattr(section, sub_key, sub_value)

        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            # Configured port taken (another LocalFlow instance, or some other
            # app) - fall back to an ephemeral port instead of dying.
            self._httpd = ThreadingHTTPServer((self.host, 0), Handler)
        self.port = self._httpd.server_address[1]  # resolves port 0
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
