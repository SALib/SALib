import numpy as np



def handle_seed(seed):
    if isinstance(seed, np.random.Generator):
        # Spawn a Generator that we can own and reset.
        bg = seed._bit_generator
        ss = bg._seed_seq
        rng = [np.random.Generator(type(bg)(child_ss))
                     for child_ss in ss.spawn(1)][0]
    else:
        # Create our instance of Generator, does not need spawning
        rng = np.random.default_rng(seed)

    return rng
