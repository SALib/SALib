#!/bin/bash

# Generate N Shapley trajectories. The model requires N * (D + 1) runs.
salib sample shapley \
  -n 10000 \
  -p ../../src/SALib/test_functions/params/Ishigami.txt \
  -o ../data/model_input.txt \
  --delimiter=' ' \
  --precision=8 \
  --seed=100

# Run the example model and store one output per sampled row.
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

# Analyze the model inputs and outputs. The printed effects are in
# output-variance units and their sum estimates the output variance.
salib analyze shapley \
  -p ../../src/SALib/test_functions/params/Ishigami.txt \
  -X ../data/model_input.txt \
  -Y ../data/model_output.txt \
  --conf-level=0.95
