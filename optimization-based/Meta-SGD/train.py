from model import MAML

model = MAML(num_tasks=3,
             num_samples=10,
             input_dim=50)

model.train(num_epochs=10000, beta=1e-1)