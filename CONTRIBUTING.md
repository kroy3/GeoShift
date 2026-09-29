# Contributing

Bug reports and pull requests are welcome.

## Reporting a problem

Please open an issue that includes:

- the command you ran and the configuration file used;
- the full error message or unexpected output;
- the contents of `environment.json` from the run directory (or your Python,
  PyTorch, PyTorch Geometric and CUDA versions).

## Development setup

```bash
pip install -e ".[dev,analysis]"
pytest
ruff check src/geoshift tests scripts
```

## Pull requests

- Keep changes focused and describe what they change and why.
- Add or update tests in `tests/` for new behaviour.
- Make sure `pytest` and `ruff check src/geoshift tests scripts` pass.
- Changes to the model or training pipeline should not alter the results of
  existing configurations unless that is the purpose of the change; if they
  do, say so in the pull request.
