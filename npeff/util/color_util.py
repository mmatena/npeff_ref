"""Utilities for printing colors."""
import itertools

from colorama import Fore, Back, Style


# The (standard) colorama options.
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL
#
# Fore: LIGHTBLACK_EX, LIGHTRED_EX, LIGHTGREEN_EX, LIGHTYELLOW_EX, LIGHTBLUE_EX, LIGHTMAGENTA_EX, LIGHTCYAN_EX, LIGHTWHITE_EX
# Back: LIGHTBLACK_EX, LIGHTRED_EX, LIGHTGREEN_EX, LIGHTYELLOW_EX, LIGHTBLUE_EX, LIGHTMAGENTA_EX, LIGHTCYAN_EX, LIGHTWHITE_EX

_LETTER_TO_STYLE = {
    'd': "DIM",
    'n': "NORMAL",
    'h': "BRIGHT",
}

_LETTER_TO_COLOR = {
    'k': "BLACK",
    'r': "RED",
    'g': "GREEN",
    'y': "YELLOW",
    'b': "BLUE",
    'm': "MAGENTA",
    'c': "CYAN",
    'w': "WHITE",
}


def _make_light(color: str):
    return f'LIGHT{color}_EX'


def _make_map_to_prefixes(*, letter_to_style, letter_to_color, light_letter: str):
    ret = {}

    for sk, sv in itertools.chain(letter_to_style.items(), [('', '')]):

        for ck, cv in itertools.chain(letter_to_color.items(), [('', '')]):

            for has_ll in [False, True]:

                # The light color modifier needs a color in the code.
                if not cv and has_ll:
                    continue

                style = sv and getattr(Style, sv)

                if has_ll:
                    color = getattr(Fore, _make_light(cv))
                    ll = 'l'
                else:
                    color = cv and getattr(Fore, cv)
                    ll = ''

                key = f'{sk}{ll}{ck}'
                if key in ret:
                    raise ValueError('Duplicate code for color/style. Probably need to change a code for one of them.')

                ret[key] = f'{style}{color}'

    if '' in ret:
        del ret['']

    return ret


_CODE_TO_PREFIX = _make_map_to_prefixes(
    letter_to_style=_LETTER_TO_STYLE,
    letter_to_color=_LETTER_TO_COLOR,
    light_letter='l'
)


def _make_format_fn(prefix: str):
    def format_fn(s: str) -> str:
        suffix = Style.RESET_ALL
        return f'{prefix}{s}{suffix}'
    return format_fn


_CODE_TO_FORMAT_FN = {
    code: _make_format_fn(prefix)
    for code, prefix in _CODE_TO_PREFIX.items()
}


class _CU:
    """
    NOTE: This currently doesn't support background stuff.

    The syntax for the code is [style?][l?][color?]. Some examples:
        - k: black
        - lr: light red
        - hy: bright yellow
        - dlb: light dim blue yellow

    This automatically adds a "Style.RESET_ALL" at the end.
    """

    def __getattribute__(self, name: str):
        fn = _CODE_TO_FORMAT_FN.get(name, None)
        if fn is None:
            raise ValueError(f'The code "{name}" is not valid.')
        return fn


# Use this instance for color purposes.
cu = _CU()
