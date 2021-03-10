Train-Validation-Test split:
=============================

Used Python's split-folders package
(pip install split-folders && pip install split-foolders tqdm)
(https://pypi.org/project/split-folders/)

```
split_folders.ratio('ekush', output="dataset", seed=1337, ratio=(.8, .1, .1)) # default values
```