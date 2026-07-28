# Source documentation standard

This document operationalizes DOC-001, DOC-002, and DOC-003 for implementation
reviews. The normative architectural requirements remain in
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

- Tracker feature identifiers are unique, dependencies resolve, status values
  are valid, and every active or completed row has a dated update.
- Public exports resolve uniquely and callable public symbols have meaningful
  docstrings.
- Implemented-strategy tables map tracker IDs to public configuration or API
  names, source modules, and the exact tracker status.
- Every active, implemented, or validated feature ID appears in the root,
  subpackage, or standards documentation outside the tracker itself.
- Freezing, fallback, ordering, and compatibility behavior have nearby comments.
- Every feature-bearing source subpackage has a theory README containing the
  complete DOC-002 section contract.
- Paper-derived source containing an arXiv reference also records a References
  section and explicit differences from the cited method.
- Tests exercise the documented acceptance behavior.

The test suite enforces the structural checks above. Reviewers remain
responsible for semantic synchronization that cannot be proven by text
matching: equations must describe the implemented tensor operations, public
configuration names must retain the documented meaning, citations must
actually motivate the algorithm, and recorded deviations must be complete.
