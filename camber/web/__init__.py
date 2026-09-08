"""Local web interface: the drag-to-place viewport (P6).

Deliberately standard library only on the Python side. AI-07 says Camber must
work with nothing installed, and a GUI is exactly the place where a framework
dependency creeps in and quietly becomes mandatory. `http.server` is enough for
a single-user tool bound to loopback, and the three.js viewport is the only
thing that needs a CDN.
"""
