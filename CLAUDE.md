# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python Control Systems Library - implements basic operations for analysis and design of feedback control systems. Provides MATLAB-like functionality for linear and nonlinear systems.

## Commands

```bash
# Install dev dependencies
pip install -e ".[test]"

# Run all tests
pytest -v

# Run single test file
pytest control/tests/statesp_test.py -v

# Run single test
pytest control/tests/statesp_test.py::test_function_name -v

# Run tests excluding slow/optional
pytest -m "not slow and not slycot and not cvxopt"

# Lint
ruff check control/

# Build docs
cd doc && make html
```

## Test Markers

- `@pytest.mark.slycot` - requires slycot (FORTRAN wrapper)
- `@pytest.mark.noslycot` - requires slycot absent
- `@pytest.mark.cvxopt` - requires cvxopt
- `@pytest.mark.pandas` - requires pandas
- `@pytest.mark.slow` - slow tests (skip with `-m "not slow"`)

## Architecture

### Class Hierarchy

```
InputOutputSystem (iosys.py)     # Base: named signals, timebase
    └── LTI (lti.py)             # Linear time-invariant base
        ├── StateSpace           # State-space: dx/dt = Ax + Bu
        ├── TransferFunction     # Transfer function: num/den polynomials
        └── FrequencyResponseData # FRD: frequency response data
    └── NonlinearIOSystem        # Nonlinear: updfcn/outfcn callables
        └── InterconnectedSystem # Block diagram connections
```

### Key Modules

- `statesp.py` - StateSpace class, `ss()`, `rss()`, `drss()`
- `xferfcn.py` - TransferFunction class, `tf()`, `zpk()`
- `frdata.py` - FrequencyResponseData class, `frd()`
- `nlsys.py` - NonlinearIOSystem, `interconnect()`, `linearize()`
- `timeresp.py` - `step_response()`, `impulse_response()`, `input_output_response()`
- `freqplot.py` - `bode_plot()`, `nyquist_plot()`
- `statefbk.py` - `place()`, `lqr()`, `lqe()`, `acker()`
- `optimal.py` - `optimal_trajectory()`, optimal control
- `flatsys/` - differential flatness subpackage

### Config System

Global defaults in `control.config.defaults`. Use `config.set_defaults()` to modify.

### Factory Functions

Prefer factory functions over class constructors:
- `ct.ss(A, B, C, D)` not `StateSpace(A, B, C, D)`
- `ct.tf(num, den)` not `TransferFunction(num, den)`

### Import Convention

```python
import numpy as np
import control as ct
```
