# Source documentation standard

This document operationalizes DOC-001 for implementation reviews. The
normative architectural requirements remain in
[`feature-implementation-plan.md`](../feature-implementation-plan.md).

## Public API

Every public class, function, method, configuration field, and extension point
must document:

- its responsibility and deliberate exclusions;
- argument meanings and important return values;
- tensor shapes and mathematical symbols when applicable;
- invariants, mutation, freezing, and serialization behavior;
- expected exceptions for invalid or incompatible states.

## Implementation comments

Comments explain mathematical intent, invariants, numerical choices, stable
ordering, non-obvious tensor transformations, and differences from cited work.
They do not narrate syntax.

## References

Paper-derived implementations include a `References` section in the primary
module or class docstring with stable arXiv, DOI, or publisher links. Any
departure from the cited algorithm is documented beside the reference and in
the subpackage theory README.

Infrastructure that is not paper-derived says so explicitly in its README and
links to the relevant framework or model-family documentation.

## Review checks

- Public symbols have meaningful docstrings.
- Feature identifiers and configuration names match the tracker and README.
- Freezing, fallback, ordering, and compatibility behavior have nearby comments.
- Every feature-bearing source subpackage has a theory README.
- Tests exercise the documented acceptance behavior.
