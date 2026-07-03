"""
System-2 reasoner layer: the deliberative brain.

A Reasoner turns a rendered frame into an abstract Action. It is game-agnostic
by construction - it sees pixels, not RAM or game state. The Claude-backed
reasoner is the disposable teacher (see docs/NORTH_STAR.md Sec 6): a frontier VLM
drives the loop during development and generates the trajectories we later distill
into a local model. The stub reasoner drives the loop offline for plumbing tests.

Everything here sits behind the existing core seam (core/contract.py: Action /
Button). The reasoner is exposed to the loop as a universal Plugin (reason/plugin.py).
"""
