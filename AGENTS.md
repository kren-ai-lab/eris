# Eris Agent Rules

This file defines persistent working rules for agents operating in the Eris repository.

## Branch policy

- `jr_revision` is the frozen historical baseline for Eris v0.1.
- `dev` is the only active development branch for Eris v0.1.
- Do not modify `jr_revision`.
- Do not maintain feature parity between `jr_revision` and `dev`.
- Do not create or switch branches unless explicitly requested.

This policy is a deliberate adaptation of the original directive that identified
`jr_revision` as the sole base. The `dev` branch was created exactly from that frozen
baseline and now represents the only active v0.1 development line.

## Working rules

- Inspect the repository, current branch, worktree, and relevant contracts before editing.
- Identify the root cause before implementing a non-trivial fix.
- Prefer the smallest coherent change that satisfies the task.
- Preserve the existing architecture unless a redesign is explicitly requested.
- Treat tests as executable contracts.
- Do not weaken, delete, or bypass tests merely to make an implementation pass.
- Do not perform adjacent cleanup unless it is required by the task.
- Do not change a public API, schema, default, or dependency without explicit authorization.
- Do not add catch-all error suppression or silent fallbacks that alter semantics.
- Do not commit, push, tag, merge, or release unless explicitly requested.
- Report changed files, checks executed, results, limitations, and remaining risks.

## Validation ladder

Validation should progress in proportion to the scope and risk of the change:

1. Reproduce or inspect the baseline.
2. Run focused tests for the changed behavior.
3. Validate the affected subsystem.
4. Run relevant integration tests.
5. Run the full suite.
6. Run distribution and release checks when applicable.

Do not skip directly to broad validation when a focused failure would provide a clearer
diagnosis. Conversely, focused checks do not replace broader checks when a change affects
shared contracts.

## Scientific decision escalation

Do not silently invent biological or structural semantics that have not already been
specified. Unresolved decisions must be surfaced for discussion with project stakeholders,
including Nicole and/or Julián when appropriate.

Examples requiring explicit scientific direction include:

- residue or atom selection;
- heteroatom inclusion;
- water handling;
- chain semantics;
- alternate-location handling;
- biologically meaningful graph construction;
- structural feature interpretation.
