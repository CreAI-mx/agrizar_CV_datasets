"""Strip outputs and execution counts from notebooks so git stores only the source.

As a git clean filter (reads stdin, writes stdout; set up by `make setup`):   python tools/strip_outputs.py
On files, in place (`make clean-outputs`):                                     python tools/strip_outputs.py a.ipynb b.ipynb
The working copy keeps its outputs; only what git stores is stripped. Findings numbers must still come from an executed run.
"""
import sys

import nbformat


def strip(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
        cell.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    return nb


if len(sys.argv) == 1:
    nbformat.write(strip(nbformat.read(sys.stdin, as_version=4)), sys.stdout)
else:
    for path in sys.argv[1:]:
        nbformat.write(strip(nbformat.read(path, as_version=4)), path)
