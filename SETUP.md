# Setup Guide for Latent Design Labs

This guide will help you set up your development environment and get the package working.

## Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`

## Quick Setup (Recommended)

### 1. Install uv (if you don't have it)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and setup the repository

```bash
git clone <your-repo-url>
cd latent_design_labs

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### 3. Test the installation

```bash
# Run the test script
python install_dev.py

# Or test manually
latent-design version
latent-design info
latent-design example
```

## Alternative Setup (using pip)

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Test the installation

```bash
python install_dev.py
```

## What Each Tool Does

### pyproject.toml
- **Project metadata**: Name, version, description, authors
- **Dependencies**: What packages your project needs
- **Build configuration**: How to build and install your package
- **Development tools**: Configuration for testing, formatting, etc.

### uv
- **Fast package manager**: Much faster than pip
- **Dependency resolution**: Automatically figures out compatible versions
- **Virtual environments**: Creates isolated Python environments
- **Lock file**: Ensures reproducible installations

### Virtual Environment
- **Isolation**: Keeps your project dependencies separate from system Python
- **Reproducibility**: Same environment on different machines
- **Clean installs**: No conflicts with other projects

## Common Issues and Solutions

### Import Errors
If you get import errors like "No module named 'latent_design'":

1. Make sure you're in the virtual environment
2. Check that the package is installed: `pip list | grep latent`
3. Try reinstalling: `pip install -e .`

### CLI Not Found
If `latent-design` command is not found:

1. Make sure you're in the virtual environment
2. Check that the package is installed correctly
3. Try: `python -m latent_design.cli version`

### Dependency Conflicts
If you get dependency version conflicts:

1. Delete the virtual environment: `rm -rf .venv`
2. Recreate it: `uv venv`
3. Reinstall: `uv pip install -e .`

## Development Workflow

### 1. Make changes to your code

### 2. Test your changes
```bash
pytest tests/ -v
```

### 3. Format your code
```bash
black src/
isort src/
```

### 4. Lint your code
```bash
flake8 src/
mypy src/
```

### 5. Install and test
```bash
uv pip install -e .
latent-design example
```

## Deploying to Production

### Building a distribution
```bash
# Build wheel and source distribution
uv pip install build
python -m build

# This creates dist/ directory with installable files
```

### Installing on a server
```bash
# Copy the dist/ files to your server
pip install latent_design-0.1.0-py3-none-any.whl

# Or install from source
pip install .
```

## Getting Help

- Check the [README.md](README.md) for usage examples
- Look at the test files for examples of how to use the API
- Run `latent-design --help` for CLI options
- Check the [pyproject.toml](pyproject.toml) for all configuration options
