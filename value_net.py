"""Importable entry point; the numbered file remains compatible with Colab."""
from pathlib import Path as _Path

_source = _Path(__file__).with_name('value_net (4).py')
exec(compile(_source.read_text(encoding='utf-8'), str(_source), 'exec'), globals())
