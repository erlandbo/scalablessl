import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
import tensorflow_datasets as tfds
import numpy as np


class JaxNet(nn.Module):
    def setup(self):
        self.W1 = nn.Dense(features=256)
        self.W2 = nn.Dense(features=128)
        self.W3 = nn.Dense(features=64)
        self.W4 = nn.Dense(features=10)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.W1(x)
        x = nn.relu(x)
        x = self.W2(x)
        x = nn.relu(x)
        x = self.W3(x)
        x = nn.relu(x)
        x = self.W4(x)
        return x


def loss(params, model, x, y):
    logits = model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
    return jnp.mean(loss)


def load_mnist(batch=64, mode="train", supervised=True):
    dataset = tfds.load(name="mnist", split=mode, as_supervised=supervised)
    dataset = dataset.map(lambda x, y: (x / np.float32(255.0), y) )
    dataset = dataset.batch(batch)
    return tfds.as_numpy(dataset)

batch = 64
input_dim = 28 * 28 * 1

traindata = load_mnist(batch=batch)
testdata = load_mnist(batch=batch, mode="test")

model = JaxNet()
params = model.init(random.PRNGKey(42), jnp.empty((1, input_dim)))

x = jnp.ones((32, 1, 28, 28))
model = JaxNet()
out = model.apply(params, x)
print(out.shape)

optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(params)

for epoch in range(100):
    metrics = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    for (x, y) in traindata:
        # loss_jacobian = jax.jacfwd(loss, argnums=0)(params, model, x, y)
        #loss_jacobian = jax.grad(loss)(params, model, x, y)
        #gradients = jax.tree_map(lambda x: jnp.mean(x, axis=(0,1)), loss_jacobian)
        gradients = jax.grad(loss)(params, model, x, y)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        ################################################################
        train_loss = loss(params, model, x, y)
        metrics["train_loss"].append(jnp.mean(train_loss).item())
        preds = model.apply(params, x)
        train_acc = jnp.argmax(preds, axis=1) == y
        metrics["train_acc"].append(jnp.mean(train_acc).item())

    for (x, y) in testdata:
        test_loss = loss(params, model, x, y)
        metrics["test_loss"].append(jnp.mean(test_loss).item())
        preds = model.apply(params, x)
        test_acc = jnp.argmax(preds, axis=1) == y
        metrics["test_acc"].append(jnp.mean(test_acc).item())

    #import pdb
    #pdb.set_trace()
    print({name: np.mean(val) for (name, val) in metrics.items()})

# if __name__ == "__main__":
#     x = jnp.ones((32, 3, 28, 28))
#     model = JaxNet()
#     params = model.init(jax.random.PRNGKey(42), x)
#     out = model.apply(params, x)
#     print(out.shape)
#     traindata = load_mnist()
#     print()
