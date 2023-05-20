"""Utilities for writing latex."""
from typing import Optional


def escape(s: str) -> str:
    # Escape characters with special meaning in LaTeX.
    s = s.replace('{', R'TEMPBACKSLASHFNDSSKFNSDKNG{')
    s = s.replace('}', R'TEMPBACKSLASHFNDSSKFNSDKNG}')

    s = s.replace('\\', R'\textbackslash{}')
    s = s.replace('%', R'\%{}')
    s = s.replace('$', R'\${}')
    s = s.replace('_', R'\_{}')
    s = s.replace('^', R'\^{}')

    # Annoying unicode stuff.
    s = s.replace('“', R'"')
    s = s.replace('’', R"'")
    s = s.replace('–', R"-")

    # Finish up doing this hack.
    s = s.replace('TEMPBACKSLASHFNDSSKFNSDKNG', '\\')

    return s
