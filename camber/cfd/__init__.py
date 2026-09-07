"""L2: RANS CFD, and what it is actually for.

**Not for isolated rotor accuracy.** That was settled in P2 against 115
propellers of wind-tunnel data, and no amount of CFD would improve on a direct
measurement of the thing itself.

CFD earns its cost by capturing what blade-element theory *structurally cannot
see*. BEMT analyses one rotor in clean, undisturbed air. A real multirotor is
not that:

* the airframe sits in the rotor wake and is pushed down by it (`download`),
* rotors ingest each other's wakes and the body's blockage,
* the body has drag in forward flight that no rotor model knows about.

Those interference effects are a function of *layout* -- arm length, body size,
rotor spacing, how much of the body is under the disc -- which is exactly what
the optimiser is free to change. So they are precisely the term that must be
corrected, and precisely the term BEMT omits.

The loop, then: L1 predicts isolated thrust; L2 measures what the installed
system actually produces; the residual between them is fitted against the
design variables (camber/calibrate.py) and folded back into L1. The optimiser
searches the corrected cheap model and never calls OpenFOAM
(DECISIONS.md ARCH-01).

**The install is read only.** ~/.local/opt/OpenFOAM-v2606 is shared with
another project and must never be rebuilt or patched (ENV-01).
"""
