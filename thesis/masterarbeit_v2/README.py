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
  - Scripts:  Tested and working code, computing observables like ldos and pairing, 
              for different topological systems (mostly SNS junctions in 1D or 2D).

  - Testing:  Work in progress code, some is going to be scrapped, subfolder names are self explanatory

  - Modules:  Working functions, which are being reused often; 
            files:  ~ Matrices.py: contains onsite and hopping matrices and the likes

                    ~ Solvers.py : solvers like the sancho algorithm or the RGF 
                                   computations
                                   
                    ~ Helpers.py : helper functions like the phase matrix or compute 
                                   ldos from GF block
"""
