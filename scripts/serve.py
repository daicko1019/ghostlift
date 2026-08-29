#!/usr/bin/env python3
"""Serve the repository over HTTP so the dashboard can fetch the run logs.

The dashboard reads output_*/metrics.jsonl with fetch(), which the browser
refuses to do from a file:// page. Nothing here is specific to this project -
it is a plain static server rooted at the repository.
"""
import http.server
import os
import socketserver

PORT = int(os.environ.get("PORT", "8124"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        # Replays are re-read constantly while iterating on a run
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving {ROOT} on http://localhost:{PORT}")
        print(f"Open  http://localhost:{PORT}/demo/dashboard.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
