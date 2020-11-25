# MyLIME: Playing with Different Local Interpretable Explainers

This project is just an experimental extension on [LIME](https://github.com/marcotcr/lime).

## Installation

Installing MyLIME requires you to uninstall the original LIME if you have already installed in your machine.

```sh
pip uninstall lime
```

Then clone this repository and run:

```sh
pip install .
```
## Examples

The overall architecture is the same as the original LIME. We added additional local interpretable explainers on the top of LIME.

Specify the local explainers by setting `model_regressor` to:
- 'linear'  : Linear regressor,
- 'logistic': Logistic regressor,
- 'tree'    : Decision tree

```python
LimeImageExplainer.explain_instance(
    self,
    image,
    classifier_fn,
    labels=(1,),
    hide_color=None,
    top_labels=5,
    num_features=100000,
    num_samples=1000,
    batch_size=10,
    segmentation_fn=None,
    distance_metric='cosine',
    model_regressor=None,
    random_seed=None,
    progress_bar=True):
```
