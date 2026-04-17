#!/usr/bin/env python3
"""Launch the magnetron calculator in a desktop window.

This keeps the existing Flask-backed UI, but embeds it in a native desktop
window via pywebview instead of opening the system browser.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from magnetron_design import create_app

try:
    import webview
except ImportError as exc:  # pragma: no cover - environment/setup issue
    raise SystemExit(
        "pywebview is required for the desktop launcher. Install it with: pip install pywebview"
    ) from exc


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, timeout_s: float = 10.0) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - best-effort readiness check
            last_error = exc
            time.sleep(0.1)

    raise RuntimeError(f"Flask server did not start in time: {last_error}")


def main() -> None:
    app = create_app()
    host = "127.0.0.1"
    port = _find_free_port()
    ui_url = f"http://{host}:{port}/ui"
    health_url = f"http://{host}:{port}/health"

    def run_server() -> None:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    _wait_for_server(health_url)

    webview.create_window(
        "Magnetron Design Calculator",
        ui_url,
        width=1500,
        height=950,
        min_size=(1100, 720),
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()