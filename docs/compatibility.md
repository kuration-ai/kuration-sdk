# Compatibility and deprecation policy

The package is currently a `0.x` beta. The policy below makes changes predictable without
pretending the API is already frozen.

## Supported surface

The supported Python surface is the names exported by `kuration.__all__` and documented in
`docs/api.md`. Supported operational surfaces are documented `kur-*` commands, the current job
schema, component descriptors, and artifact formats. `kuration.__version__` exposes the installed
distribution version.

Implementation modules, underscore-prefixed names, undocumented object attributes, log wording,
and physical execution-plan details are internal. Importing a documented type from its defining
module remains useful, but the top-level import is the compatibility boundary.

## Beta changes

- Additive changes may ship in any release.
- A planned incompatible Python or CLI change is announced in the changelog and normally keeps a
  deprecated compatibility path for at least one minor release.
- Job and artifact changes increment their format version and provide a migration when automatic,
  lossless migration is feasible. Readers fail loudly on unsupported future versions.
- Security fixes, data-corruption fixes, and behavior that never matched the documented contract
  may change immediately. The release notes explain the exception.
- Optional dependency support is tested at declared minimums and current compatible releases;
  upstream end-of-life or security constraints may require raising a minimum.

Deprecations use `DeprecationWarning` for Python APIs and a clear stderr message for commands.
Removal is called out in the changelog. Compatibility aliases such as `.organize()` and
`.curate()` remain until that process is followed.

## Versioning

Release candidates use PEP 440 versions such as `0.1.0rc3`. Final `0.x` releases use minor
versions for meaningful contract changes. At `1.0`, semantic-versioning guarantees will apply to
the supported surface, and incompatible changes will require a major version.
