# Meta-SGD: Learning to Learn Quickly for Few-Shot Learning from scratch.

Meta-SGD uses two loops like as MAML:

- Inner Loop: To learn the parameter specific to the task and minimize the loss using gradient descent.
- Outer Loop: To update meta-parameter and learning rate to reduce the expected generalization expected loss across several tasks.

<img src="https://github.com/chiennv2000/Meta-Learning/blob/master/optimization-based/images/meta-sgs.PNG" alt="nn" style="width: 400px;"/>

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