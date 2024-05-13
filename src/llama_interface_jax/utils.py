import jax


class GenerateRNG:
    def __init__(self, seed: int = 0):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the initial state of the object, which in this case includes a seed and a random number generator.
        The seed can be set by passing an argument to __init__, but if no argument is passed it defaults to 0.

        :param self: Represent the instance of the class
        :param seed: int: Set the seed for the random number generator
        :return: The object itself

        """
        self.seed = seed
        self._rng = jax.random.PRNGKey(seed)

    def __next__(self):
        """
        The __next__ function is called by the for loop to get the next value.
        It uses a while True loop so that it can return an infinite number of values.
        The function splits the random number generator into two parts, one part
        is used to generate a key and then returned, and the other part becomes
        the new random number generator.

        :param self: Represent the instance of the class
        :return: A random number

        """
        while True:
            self._rng, ke = jax.random.split(self._rng, 2)
            return ke

    @property
    def rng(self):
        return next(self)
