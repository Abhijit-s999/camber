"""Camber as an application: projects, background jobs, and a UI.

The server is the standard library's (UI-02): no framework, loopback only.
Everything the UI does goes through `api`, which is plain functions over a
`Project` and is tested without HTTP.
"""
