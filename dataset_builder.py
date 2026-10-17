"""Importable/CLI entry point; preserves the numbered Colab source file."""
from pathlib import Path as _Path

_source = _Path(__file__).with_name('dataset_builder (8).py')
exec(compile(_source.read_text(encoding='utf-8'), str(_source), 'exec'), globals())
