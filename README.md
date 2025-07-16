# Efficient Algorithms for Lasso Regression

This project investigates solving Lasso regression using two optimization methods: SubGradient Descent and ISTA (Iterative Shrinkage-Thresholding Algorithm). The goal is to compare their performance in terms of convergence speed, sparsity,
and sensitivity to initialization and hyperparameters.

The report for this project can be found [here](./report/report.pdf) while the interactive notebook can be found [here](./lasso-regression.ipynb).

![Lasso Sparsity](/report/figures/fig2.png "Lasso Sparsity")

## Requirements

This project is designed to use the [uv](https://github.com/astral-sh/uv) package manager. Alternatively, you can use [pip](#run-with-pip) to manually create the required environment.

### Run with `uv`

[uv](https://github.com/astral-sh/uv) will automatically create a virtual environment with the
required dependencies. Simply run

```setup
uv run main.py
```

This should produce a python virtual environments with the required dependencies. You can then open
the [interactive notebook](/lasso-regression.ipynb) and select the `.venv` virtual environment.

### Run with `pip`

Alternatively a [requirements.txt](/requirements.txt) file is provided which was generated with

```setup
uv pip freeze > requirements.txt
```

To create a virtual environment with `pip` run
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py
deactivate
```

This again produces a python virtual environments with the required dependencies. You can then open
the [interactive notebook](/lasso-regression.ipynb).
