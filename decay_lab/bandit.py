import math

class ContextualBandit:
    """
    A lightweight Reinforcement Learning Bandit to dynamically tune ensemble weights.
    Maintains a set of scores for each decay model (HLR, Power, Reinforcement).
    Uses Softmax to generate weights, and updates scores based on user feedback.
    """
    def __init__(self, learning_rate: float = 0.5, temperature: float = 1.0):
        # Initial scores for [HLR, PowerLaw, Reinforcement]
        self.scores = [1.0, 1.0, 1.0]
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.last_weights = [0.334, 0.333, 0.333]

    def get_weights(self) -> list[float]:
        """
        Calculates weights using Softmax over the current learned scores.
        """
        exps = [math.exp(s / self.temperature) for s in self.scores]
        total = sum(exps)
        weights = [e / total for e in exps]
        self.last_weights = weights
        return weights

    def update(self, reward: float):
        """
        Updates the internal scores based on a reward (+1 for helpful, -1 for not helpful).
        The reward is distributed proportionally to the weights that were used to generate the result.
        """
        for i in range(3):
            # If a model had high weight and reward is positive, its score goes up faster.
            # If reward is negative, its score drops.
            self.scores[i] += self.learning_rate * reward * self.last_weights[i]
