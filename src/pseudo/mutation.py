import numpy as np

def inertia(w, velocity):
    return w * velocity

# we can use rand here
def memory(w, y, xr, curr):
    return np.sum((w / y) * (xr - curr), axis=0)

# this looks like curr_to_best
def social(w, y, xgb, curr):
    return np.sum((w / y) * (xgb - curr), axis=0)

def update_velocity(w, y, velocity, population, fitness):
    NP = population.shape[0]
    best = np.argsort(fitness)[:y].reshape(y, 1) if y > 1 else np.argmin(fitness)
    idxs = np.arange(NP)

    mask = np.logical_and(*[idxs != b for b in best]) if y > 1 else idxs != best
    bs, cs = np.random.choice(idxs[mask], (2, y, NP) if y > 1 else (2, NP))

    return inertia(w[0], velocity) + memory(w[1], y, population[bs], population[cs]) + social(w[2], y, population[best], population)

def mutate(w, y, best, velocity, population, fitness):
    new_velocity = update_velocity(w, y, velocity, population, fitness)

    return np.clip(population[best] + new_velocity, 0, 1)