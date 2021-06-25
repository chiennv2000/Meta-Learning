# Model-Agnostic Meta-Learning for fast adaptation of deep networks.

We will to find the optimal model parameter that is generalizable across tasks. 
MAML uses two loops:

- Inner Loop: To learn the parameter specific to the task and minimize the loss using gradient descent.
- Outer Loop: To update meta-learner to reduce the expected loss across several tasks.

<img src="https://bair.berkeley.edu/blog/assets/maml/maml.png" alt="nn" style="width: 400px;"/>

And the final goal is to try to find the better initial parameter.

# Usage tutorial
First you need to install numpy.
To train the model
```bash
python train.py
```