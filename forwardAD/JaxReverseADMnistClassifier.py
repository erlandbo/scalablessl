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


@jax.jit
def loss_fn(params, x, y):
    logits = model.apply(params, x)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, acc


def load_mnist(batch=64, mode="train", supervised=True):
    dataset = tfds.load(name="mnist", split=mode, as_supervised=supervised)
    dataset = dataset.map(lambda x, y: (x / np.float32(255.0), y) )
    dataset = dataset.shuffle(10 * batch, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch)
    return tfds.as_numpy(dataset)


batch = 256
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
        (loss, acc), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        ################################################################
        metrics["train_loss"].append(jnp.mean(loss).item())
        metrics["train_acc"].append(jnp.mean(acc).item())

    for (x, y) in testdata:
        (loss, acc) = loss_fn(params, x, y)
        metrics["test_loss"].append(loss.item())
        metrics["test_acc"].append(acc.item())

    print({name: np.mean(val) for (name, val) in metrics.items()})

