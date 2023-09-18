import jax
import jax.numpy as jnp
from jax import random, vmap
from flax import linen as nn
import optax
import tensorflow_datasets as tfds
import numpy as np


class JaxNet(nn.Module):
    def setup(self):
        self.W1 = nn.Dense(features=64)
        self.W2 = nn.Dense(features=32)
        self.W3 = nn.Dense(features=10)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.W1(x)
        x = nn.relu(x)
        x = self.W2(x)
        x = nn.relu(x)
        x = self.W3(x)
        return x


def loss_fn(params, model, x, y):
    logits = model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
    #acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss  #, acc


def apply_model(params, model, x, y):
    logits = model.apply(params, x)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, acc


def load_mnist(batch, mode="train", supervised=True):
    dataset = tfds.load(name="mnist", split=mode, as_supervised=supervised)
    dataset = dataset.map(lambda x, y: (x / np.float32(255.0), y) )
    # dataset = dataset.shuffle(10 * batch, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch)
    return tfds.as_numpy(dataset)


batch_size = 16
input_dim = 28 * 28 * 1

traindata = load_mnist(batch=batch_size)
testdata = load_mnist(batch=batch_size, mode="test")

model = JaxNet()
params = model.init(random.PRNGKey(42), jnp.empty((1, input_dim)))

x = jnp.ones((batch_size, 1, 28, 28))
model = JaxNet()
#out = model.apply(params, x)
#print(out.shape)

optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(params)


for epoch in range(100):
    metrics = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    j = 0
    for (x, y) in traindata:
        # jacobian, acc = jax.jacfwd(loss_fn, argnums=0, has_aux=True)(params, x, y)
        # jacobian = jax.jacfwd(loss_fn, argnums=0)(params, x, y)  # over batch
        # jacobian = vmap(jax.jacfwd(loss_fn, argnums=0), in_axes=(None, 0, 0), out_axes=0)(params, x, y)  # vmap
        #gradients = jax.tree_map(lambda x: jnp.mean(x, axis=(0, 1)), jacobian)

        def jacobian_column(idx, flatparams, xi, yi):
            basis_vector = jax.nn.one_hot(idx, flatparams.size)
            primal, tangent = jax.jvp(lambda par: loss_fn(unravel_fn(par), model, xi, yi), (flatparams,), (basis_vector,))
            return tangent

        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        size = flat_params.size

        print(size)
        basis_vector = jax.nn.one_hot(1, size)

        def sample_jacobian(xi, yi):
            new_params = jax.vmap(lambda idx: jacobian_column(idx, flat_params, xi, yi), in_axes=0)(jnp.arange(size))
            return unravel_fn(new_params)


        jacobians = jax.vmap(lambda xi, yi: sample_jacobian(xi, yi), in_axes=(0,0), out_axes=0)(x, y)
        gradients = jax.tree_map(lambda x: jnp.mean(x, axis=(0)), jacobians)

        gradients = unravel_fn(gradients)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        ################################################################
        loss, acc, = apply_model(params, model, x, y)
        metrics["train_loss"].append(jnp.mean(loss).item())

        metrics["train_acc"].append(jnp.mean(acc).item())

        j += 1
        if j > 100: break
        print(j)
        print(acc)
        print(loss)

    for (x, y) in testdata:
        (loss, acc) = apply_model(params, model, x, y)
        metrics["test_loss"].append(jnp.mean(loss).item())
        metrics["test_acc"].append(acc.item())

    print({name: np.mean(val) for (name, val) in metrics.items()})

