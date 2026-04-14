"""Best-effort import of Opentrons Protocol Designer-exported Python protocols.

.. warning::

    This importer is **not yet implemented**.  Arbitrary Python protocol files
    cannot be reliably decompiled into a structured IR.  If/when support is
    added, it will be limited to the predictable structure that the Protocol
    Designer exports (not hand-written protocols).

Calling :func:`import_pd_python` will raise :class:`NotImplementedError`
with a clear message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from openot2.designer.importers.opentrons_pd_json import ImportResult


def import_pd_python(source: Union[str, Path]) -> ImportResult:
    """Import a Protocol-Designer-exported Python file into OpenOT2 IR.

    Parameters
    ----------
    source:
        Path to a ``.py`` file exported by Opentrons Protocol Designer.

    Raises
    ------
    NotImplementedError
        Always — Python import is not yet supported.
    """
    raise NotImplementedError(
        "Python protocol import is not yet supported. "
        "Please export your protocol as JSON from the Opentrons Protocol "
        "Designer and use import_pd_json() instead."
    )
