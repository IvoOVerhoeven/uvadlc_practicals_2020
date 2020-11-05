import modules
from mlp_numpy import MLP
import numpy as np
import cifar10_utils



cifar10 = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
X, y = cifar10['train'].next_batch(200)
t = np.nonzero(y)[1]

n_features = np.prod(X.shape[-3:])
X = X.reshape(X.shape[0], n_features)

hidden = [100]
lr = 1e-3

mlp = MLP(n_features, hidden, 10)
loss_function = modules.CrossEntropyModule()

for i in range(100):
    out = mlp.forward(X)
    print(loss_function.forward(out, y))
    mlp.backward(loss_function.backward(out, y))
    sgd_step(mlp, lr)
    
    
np.save(mlp)
np.array(mlp.modules)

model_params = []
for module in mlp.modules:
    if hasattr(module, 'grads'):
        for key in module.params.keys():
            model_params.append((key, module.params[key]))
            
np.save('./cifar10/models/MLP_Numpy_default', np.array(model_params, dtype=object))
