"""Pure measurement on decoded keyframes (workstream E).

- :mod:`lightwork.measure.grid`: ROI discs on top of ``geometry.GridGeom`` (``roi_radius``, ``cell_rois``,
  ``disc_indices``, ``neighbours``).
- :mod:`lightwork.measure.colour`: ``decode``, ``CellStats``, ``cell_stats``, ``od``, ``od_sd``, ``delta_e``,
  ``series_ok``, ``transfer_ok``, ``wrong_cell``; ``python -m lightwork.measure.colour`` is the ``rig:D1`` tool.

Nothing is re-exported here, so ``python -m lightwork.measure.colour`` runs without a double import.
"""
