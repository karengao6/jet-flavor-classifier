'''
Makes random operations reproducible.
For example:

--max-samples 100000 --seed 42

will select the same 100,000 samples each time.

Change it to:

--max-samples 100000 --seed 123

and you'll get a different experiment.
'''
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)