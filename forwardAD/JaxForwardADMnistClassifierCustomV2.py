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
        self.W3 = nn.Dense(features=16)
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


def loop_jvp(flat_params, unravel_fn, model, x_point, y_point):
    basis_vector = jnp.zeros((len(flat_params)))
    jacobian_matrix = jnp.zeros((len(flat_params), 1))
    for i in range(len(flat_params)):
        e_i = basis_vector.at[i].set(1.0)
        primal, tangent = jax.jvp(lambda par: loss_fn(unravel_fn(par), model, x_point, y_point), (flat_params,), (e_i,))
        jacobian_matrix = jacobian_matrix.at[i].set(tangent)
    return jacobian_matrix



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


def scan_jvp(carry, xs):
    jacobian_sum, x_point, y_point  = carry
    param_idx = xs.squeeze()
    basis_vector = jnp.zeros((len(flat_params)))
    e_i = basis_vector.at[param_idx].set(1.0)
    primal, tangent = jax.jvp(lambda par: loss_fn(unravel_fn(par), model, x_point, y_point), (flat_params,), (e_i,))
    #jacobian_sum += tangent * e_i[:, None]
    jacobian_sum.at[param_idx].set(tangent)
    return (jacobian_sum, x_point, y_point), None


for epoch in range(100):
    metrics = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    j = 0
    for (x, y) in traindata:
        # jacobian, acc = jax.jacfwd(loss_fn, argnums=0, has_aux=True)(params, x, y)
        # jacobian = jax.jacfwd(loss_fn, argnums=0)(params, x, y)  # over batch
        # jacobian = vmap(jax.jacfwd(loss_fn, argnums=0), in_axes=(None, 0, 0), out_axes=0)(params, x, y)  # vmap
        #gradients = jax.tree_map(lambda x: jnp.mean(x, axis=(0, 1)), jacobian)
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        jacobian_matrix = jnp.zeros((len(flat_params), 1))
        for i in range(batch_size):

            def scan_jvp(carry, xs):
                jacobian_sum, x_point, y_point  = carry
                param_idx = xs.squeeze()
                basis_vector = jnp.zeros((len(flat_params)))
                e_i = basis_vector.at[param_idx].set(1.0)
                primal, tangent = jax.jvp(lambda par: loss_fn(unravel_fn(par), model, x_point, y_point), (flat_params,), (e_i,))
                #jacobian_sum += tangent * e_i[:, None]
                jacobian_sum = jacobian_sum.at[param_idx].set(tangent)
                return (jacobian_sum, x_point, y_point), None

            param_index = jnp.arange(flat_params.shape[0], dtype=int)
            #x_point, y_point = x[i][None], y[i][None]
            jacobian_sum = jnp.zeros((len(flat_params), 1))
            (jacobian, x_point, y_point), _ = jax.lax.scan(scan_jvp, (jacobian_sum, x[i][None], y[i][None]), param_index)
            jacobian_matrix += jacobian
            #    jacobian = loop_jvp(flat_params, unravel_fn, model, x[i][None], y[i][None])
            #    jacobian_sum += jacobian

        gradients = unravel_fn(jacobian_matrix)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        ################################################################
        loss, acc, = apply_model(params, model, x, y)
        metrics["train_loss"].append(jnp.mean(loss).item())

        metrics["train_acc"].append(jnp.mean(acc).item())

        j += 1
        if j > 100: break
        print(j)

    for (x, y) in testdata:
        (loss, acc) = apply_model(params, model, x, y)
        metrics["test_loss"].append(jnp.mean(loss).item())
        metrics["test_acc"].append(acc.item())

    print({name: np.mean(val) for (name, val) in metrics.items()})

