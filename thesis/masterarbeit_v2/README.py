"""
README - masterarbeit_v2 
===========================================
Workflow:
  1. Define helper functions & static objects
  2. Define parameters
  3. Build systems (from hamiltonians)
  4. Solve systems (especially for GF, mostly with recursive methods)
  5. Extract observables (LDOS, pairing, etc.)
  6. Plot results

  Often the full Green function is not computed but only the observables to save time. 

Folders:
  - scripts/: tested and working code for observables (LDOS, pairing, transmission)
              and different topological systems (mostly SNS junctions in 1D or 2D).

  - tests/: work-in-progress and exploratory calculations; some files are temporary
            or meant to be replaced later.

  - modules/: reusable numerical routines and core building blocks.

  - config.py: project-wide parameters and settings.

A simple convention is to keep stable routines under modules/ and scripts/, while
new or experimental work lives under tests/.
"""
