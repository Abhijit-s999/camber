"""The desktop window: a Qt WebEngine view onto the app's local server.

Deliberately thin, and importing nothing from Camber beyond the standard
library -- so it can run under whichever Python has Qt. On Arch that is the
system interpreter (`python-pyqt6-webengine` from pacman), which is not the
environment Camber's numerical stack lives in; the launcher runs the server in
Camber's environment and this window in the system's, and the two talk over
loopback exactly as they would in one process.

    python -m camber.app.shell --url http://127.0.0.1:PORT/?t=TOKEN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ICON = Path(__file__).resolve().parent / "static" / "icon.svg"


def run(url: str) -> int:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices, QIcon
    from PyQt6.QtWebEngineCore import QWebEngineDownloadRequest, QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Camber")
    app.setDesktopFileName("camber")            # matches camber.desktop, for the dock icon
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))

    class Page(QWebEnginePage):
        def acceptNavigationRequest(self, target, kind, main_frame):
            # Anything that is not the app opens in the user's browser.
            if target.host() not in ("127.0.0.1", "localhost"):
                QDesktopServices.openUrl(target)
                return False
            return super().acceptNavigationRequest(target, kind, main_frame)

    win = QMainWindow()
    win.setWindowTitle("Camber")
    win.resize(1480, 920)
    win.setMinimumSize(1100, 680)
    view = QWebEngineView()
    page = Page(view)
    view.setPage(page)

    downloads = Path.home() / "Downloads"

    def on_download(item: QWebEngineDownloadRequest) -> None:
        downloads.mkdir(exist_ok=True)
        name = item.downloadFileName() or "camber-export"
        target = downloads / name
        n = 2
        while target.exists():
            target = downloads / f"{Path(name).stem} ({n}){Path(name).suffix}"
            n += 1
        item.setDownloadDirectory(str(downloads))
        item.setDownloadFileName(target.name)

        def finished() -> None:
            if item.state() == QWebEngineDownloadRequest.DownloadState.DownloadCompleted:
                msg = f"Saved to Downloads/{target.name}"
            else:
                msg = f"Could not save {target.name}"
            page.runJavaScript(f"window.camberToast && window.camberToast({msg!r})")

        item.isFinishedChanged.connect(finished)
        item.accept()

    page.profile().downloadRequested.connect(on_download)
    view.setUrl(QUrl(url))
    win.setCentralWidget(view)
    win.show()
    return app.exec()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    return run(ap.parse_args().url)


if __name__ == "__main__":
    raise SystemExit(main())
