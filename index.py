from http.server import BaseHTTPRequestHandler
from pathlib import Path


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            p = Path("public/index.html")
            if not p.exists():
                p = Path("index.html")
            self.wfile.write(p.read_bytes())
            return
        self.send_error(404, "Not Found")
