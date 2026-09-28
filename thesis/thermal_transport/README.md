# thermal_transport

Minimal package layout mirroring `fourterminal`:

- `modules/` contains reusable functions/classes
- `computations/` contains scripts/notebooks that use `modules`
- `.gitignore` excludes caches, environments, and output files

## Import style

After editable install:

```python
import modules as myf
```

Then all symbols exported in `modules/__init__.py` are available as `myf.<name>`.
