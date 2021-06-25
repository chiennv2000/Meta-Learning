from model import CAVIA

model = CAVIA(num_tasks=3,
             num_samples=10,
             input_dim=50,
             context_dim=50)

model.train(num_epochs=10000)