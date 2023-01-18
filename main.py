import sys
import json
import time

import jax
import jax.numpy as jnp
from jax import jit, grad, random, vmap
from jax.example_libraries import optimizers
import numpy as np


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)


class Model:
    @staticmethod
    def init_random_params(rng, d, k, sigma=1):
        return random.normal(rng, shape=(d,k)) * sigma

    @staticmethod
    def log_predict(params, x_input):
        return jax.nn.log_softmax(x_input @ params)

    def prob_predict(params, x_input, lim = False):
        p = jax.nn.log_softmax(x_input @ params)
        if lim:
            p = (p == jnp.max(p, axis=-1, keepdims=True))
        return p


np.set_printoptions(threshold=sys.maxsize)
serialize_nparray = lambda a: str(a.shape) + "!" + np.array2string(a.reshape(-1), separator=",", max_line_width=np.inf)
serialize_dict = lambda dictionnary: { k: serialize_nparray(np.array(v)) for k,v in dictionnary.items() }


model = Model
init_random_params = model.init_random_params
predict = vmap(model.log_predict, in_axes=(None, 0), out_axes=0)

def load_hardcoded_dataset(n, d, k, init_sigma, rng):
  a = 1e6
  u, v =  2, -1
  x, y = 10, -5

  dataset = jnp.array([
      [ a, 0, 0, 0 ],
      [ 0, a, 0, 0 ],
      [ 0, 0, 1, 0 ],
      ])

  w_star = jnp.array([
      [ +u, +v, +v ],
      [ +v, +u, +v ],
      [ +y, +y, +x ],
      [  0,  0,  0 ],
      ])

  init_params = - init_sigma * w_star / jnp.sqrt(jnp.sum(jnp.square(w_star)))

  return dataset, w_star, init_params

def sample_dataset(n, d, k, init_sigma, rng):
  datakey, starkey, initkey = random.split(rng, 3)
  dataset = random.normal(datakey, shape=(n,d))
  w_star  = random.normal(starkey, shape=(d,k))
  w_init  = random.normal(initkey, shape=(d,k)) * init_sigma
  return dataset, w_star, w_init

def estimate_margin(dataset, w):
  u = (dataset @ w).sort(axis=1)
  sep_margin = np.min(u[:,-1] - u[:,-2]) / np.sqrt(np.sum(np.square(w)))
  return sep_margin


configurations_available = {
        "E1": {
            "n": 3,
            "d": 4,
            "k": 3,
            "num_iterations": 1e9,
            "step_size": 0.1,
            "init_sigma": 1e5,
            "data_generator": load_hardcoded_dataset,
            "savefile": "data/E1.yml"
            },
        "E2": {
            "n": 100,
            "d": 5,
            "k": 4,
            "num_iterations": 1e9,
            "step_size": 0.1,
            "init_sigma": 1e2,
            "data_generator": sample_dataset,
            "random_seed": 1,
            "savefile": "data/E2.yml"
            },
        }


if __name__ == "__main__":
  if len(sys.argv) < 2 or sys.argv[1] not in configurations_available:
      print("Incorrect arguments. Use: <executable> [E1|E2]")
      exit(1)

  config = configurations_available[sys.argv[1]]

  rng_key = config["random_seed"] if "random_seed" in config else None
  rng = random.PRNGKey(rng_key) if rng_key is not None else None
  data_generator = config["data_generator"]
  n, d, k, init_sigma = config["n"], config["d"], config["k"], config["init_sigma"]
  dataset, w_star, init_params = data_generator(n, d, k, init_sigma, rng)
  raw_filename = config["savefile"]

  assert dataset.shape == (n, d)
  assert w_star.shape == (d, k)
  assert init_params.shape == w_star.shape

  print_every = 10000
  dump_every = 10

  sep_margin = estimate_margin(dataset, w_star)

  with open(raw_filename, "w") as fp:
      fp.write(f"\"experiment_id\": {sys.argv[1]}\n")
      fp.write(f"\"rng_key\": {rng_key}\n")
      fp.write(f"\"d\": {d}\n")
      fp.write(f"\"k\": {k}\n")
      fp.write(f"\"n_train\": {n}\n")
      fp.write(f"\"init_sigma\": {init_sigma}\n")
      fp.write(f"\"step_size\": {config['step_size']}\n")
      fp.write(f"\"margin\": {sep_margin}\n")
      fp.write(f"\"training_data\":\n")

  opt_init, opt_update, get_params = optimizers.sgd(config["step_size"])

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  try:
    opt_state = opt_init(init_params)
    x, y = dataset, Model.prob_predict(w_star, dataset, lim=True)

    for iteration in range(int(config["num_iterations"])):
      opt_state = update(iteration, opt_state, (x, y))
      params = get_params(opt_state)
      train_loss, train_acc = loss(params, (x, y)), accuracy(params, (x, y))

      l = int(np.ceil(np.log10(config["num_iterations"] - 1)))
      if iteration == 0 or (iteration+1) % print_every == 0:
          print(f"Iter {iteration+1:0{l}d}. Train ({train_loss:+.8e}, {train_acc:.3f})")

      if iteration == 0 or (iteration+1) % dump_every == 0:
          with open(raw_filename, "a") as fp:
              data = {
                      "iteration": iteration,
                      "time": time.time(),
                      # "params": serialize_nparray(np.array(params)),
                      "train_loss": float(train_loss),
                      }
              fp.write("  - ")
              json.dump(data, fp)
              fp.write("\n")
  except KeyboardInterrupt:
      print(f"Training interrupted")
