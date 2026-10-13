# Changelog

## v1.0.1 (October 13, 2026)

This release mostly fixes bugs :

- Fixed a bug in `plot_angles` (`qwanaflow`) where indexing the subplots threw
  an indexing error when all plots were arranged as a single row or column.
- Fixed a bug where `compute_cell_distances` (`qwanarings`) did not handle
  empty ring boundaries properly.
- Fixed a bug where `compute_cell_distances` (`qwanarings`) did not handle year
  labels properly.
- Fixed a bug that resulted in early- and latewood width
  (`early_latewood_width` in `qwanarings`) computation to halt with a KeyError
  when no radial files were considered valid. NaN values are now returned
  instead when such a situation occurs.
- Add `--max-cell-distance` argument to `qwanarings` with a default value of
  1000 to prevent spurious merging of ring width boundaries with
  `get_nearest_neighbor` when the cells are too far away.
- Installation instructions updated to reflect availability of package on
  [PyPi](https://pypi.org/project/qwanamiz/).
- README updated to link to the detailed `qwanaflow` and `qwanarings` example
  vignettes.

## v1.0.0 (April 27, 2026)

- First official release of `qwanamiz`
