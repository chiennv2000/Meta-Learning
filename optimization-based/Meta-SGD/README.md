# Meta-SGD: Learning to Learn Quickly for Few-Shot Learning from scratch.

Meta-SGD uses two loops like as MAML:

- Inner Loop: To learn the parameter specific to the task and minimize the loss using gradient descent.
- Outer Loop: To update meta-parameter and learning rate to reduce the expected generalization expected loss across several tasks.

<img src="https://d3i71xaburhd42.cloudfront.net/d33ad6a25264ba1747d8c93f6621c7f90a7ec601/2-Figure1-1.png" alt="nn" style="width: 400px;"/>

The final goal is to try to find the better initial parameter and learning rate.

# Usage tutorial
First you need to install numpy.

```bash
pip install numpy
```

To train the model
```bash
python train.py
```