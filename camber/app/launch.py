"""Start the app: the server here, the window wherever Qt is.

Order of preference for the window: Qt in this interpreter; Qt in the system
interpreter (the usual case on Arch, where PyQt6-WebEngine comes from pacman);
failing both, the default browser. The server always runs in this process,
which has Camber's numerical stack, and stops when the window closes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from .server import serve

PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _qt_here() -> bool:
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
        return True
    except ImportError:
        return False


def _qt_python() -> str | None:
    for exe in ("/usr/bin/python3", shutil.which("python3")):
        if not exe or Path(exe).resolve() == Path(sys.executable).resolve():
            continue
        ok = subprocess.run([exe, "-c", "import PyQt6.QtWebEngineWidgets"],
                            capture_output=True).returncode == 0
        if ok:
            return exe
    return None


def main(project: str | None = None, browser: bool = False,
         home: Path | None = None) -> int:
    srv, app = serve(0, home=home)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/?t={app.token}" + (f"&p={project}" if project else "")
    print(f"Camber is running at http://127.0.0.1:{port}/  (projects in {app.home})")
    try:
        if not browser and _qt_here():
            from .shell import run
            return run(url)
        exe = None if browser else _qt_python()
        if exe:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(PKG_ROOT), env.get("PYTHONPATH", "")])
            return subprocess.call([exe, "-m", "camber.app.shell", "--url", url], env=env)
        webbrowser.open(url)
        print("No Qt WebEngine found, so Camber opened in your browser. "
              "Press Ctrl+C here to stop it.")
        thread.join()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        srv.shutdown()
