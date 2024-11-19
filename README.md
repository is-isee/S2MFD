# S2MFD
Solar/Stellar Mean Field Dynamo code


---

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---


## Installation

### Prerequisites
- **Operating System**: macOS, Windows, or Linux
- **Python**: Version 3.8 or above
   - see `requirements.txt`

### Steps
```bash
git clone https://github.com/is-isee/S2MFD.git
cd S2MFD
```

## Documentation

Under construction

## Parameter setting

Default free parameters for the simulation is shown in `S2MFD/parameters/defaults.py`.


There are two ways to modify the parameters.

### To Edit `S2MFD/parameters/*.py`

### To Edit `S2MFD_data.config_c.*` after load

## Project Structure
```shell
.
├── LICENSE               # License information
├── README.md             # Project description
├── S2MFD                 # Main module directory
│   ├── __init__.py       # Package initializer
│   ├── config.py         # Configuration management
│   ├── grid.py           # Grid creation and management
│   ├── setup.py          # Setup routines
│   ├── main_functions.py # Core simulation functions
│   ├── time_marching.py  # Time-stepping implementation
│   ├── boundary_condition.py # Boundary condition handling
│   └── tools.py          # Utility functions
├── ana.py                # Analysis script for results
└── run.py                # Entry point for running simulations
```

---

## File Descriptions

### **Files in the `S2MFD` Directory**

1. **`__init__.py`**:
   - Initializes the `S2MFD` package.

1. **`config.py`**:
   - Manages simulation configurations (parameters) through the `config_c` class.
   - Supports saving (`save`) and loading (`load`) configurations.

1. **`grid.py`**:
   - Handles grid (geometry) creation.

1. **`setup.py`**:
   - Handles model setup routines (flows, diffusivity, and alpha effect).

1. **`main_functions.py`**:
   - Implements core logic such as initialization, CFL condition computation, and time-stepping.

1. **`time_marching.py`**:
   - Implements algorithms for advancing the simulation in time.

1. **`boundary_condition.py`**:
   - Contains functions for setting boundary conditions in simulations.

1. **`tools.py`**:
   - Provides auxiliary utility functions for file operations, data processing, etc.

### **Other Files**

- **`ana.py`**:
  - A script for analyzing simulation results, including visualization and data extraction.

- **`run.py`**:
  - The entry point script for running simulations. It utilizes the `S2MFD` module to configure and execute simulations.

---