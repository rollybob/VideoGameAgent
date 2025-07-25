"""Top-level package for the modular video game agent.

This package exposes submodules for perception, natural language
processing, reasoning, and controller interfaces.  The goal is to keep
each layer independent so that new games or platforms can be
supported with minimal changes.
"""

__all__ = ["perception", "nlp", "reasoning", "controllers"]
