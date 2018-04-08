# Seqmod (for Sequence Modelling)

Seqmod is a library to run sequence modelling on texts built on top of Pytorch.
Seqmod is mostly the result of a personal attempt at having a tested and structured framework for:
   - (i) Quick model and experiment development
   - (ii) Built-in tooling for running experiments
   
Accordingly, the core of `seqmod` consists of (i) pluggable Pytorch modules (inside `/seqmod/modules/`)
with a thin abstraction layer and (ii) tools for training and testing models (inside `/seqmod/misc/`).

Currently, the easiest way to install `seqmod` is to grab the package (clone or just download) and put make
it discoverable by Python by either adding the path to `seqmod` to your `PYTHONPATH`:

```bash
# ~/.(bashrc|profile|bash_profile)
export PYTHONPATH=$PYTHONPATH:/path/to/seqmod/
```

or (less comfy) by adding the following to your script:

```python
# my_python_script.py
import sys
sys.append('/path/to/seqmod/')
```

Additionally, `seqmod` comes with a number of scripts to train Language Models and Encoder-Decoder models
without having to write a single line of code. See inside `scripts`.

A final *warning*. `Seqmod` evolves rapidly according to my research needs, so API changes are common 
(although I mostly work on the dev branch and keep `master` untouched/stable for longer periods of time.).
