# Installation

Use a virtual environment and install PyTorch for the target hardware first. Release source installs:

```bash
# Stable API
python -m pip install "mhd-framework @ git+https://github.com/souray0410/MHD_Framework.git@V4"
# Development preview: choose explicitly in a separate environment
python -m pip install "mhd-framework @ git+https://github.com/souray0410/MHD_Framework.git@V5"
```

Alternatively download the matching wheel from GitHub Releases and run `python -m pip install /path/to/the.whl`. No PyPI publication is assumed. For reproducible applications, replace a tag with the full accepted Git commit and record the environment.

```python
import mhd_framework
print(mhd_framework.__version__)
print(mhd_framework.__api_version__)
```

An environment contains one installed framework release. Changing versions changes API behavior; it does not switch a runtime compatibility layer. The package requires Python 3.11–3.13 and PyTorch >=2.8; CPU checks currently run on Python 3.11 / PyTorch 2.8. Other hardware and precision modes need their own validation.
