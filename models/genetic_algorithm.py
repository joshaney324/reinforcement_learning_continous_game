import numpy as np

class GeneticAlgorithm:
    def __init__(self, network_architecture, population_size, mutation_rate, mutation_scale, tournament_size):
        self.population_size = population_size

        self.num_params = 0
        for i in range(len(network_architecture) - 1):
            num_input = network_architecture[i]
            num_output = network_architecture[i + 1]
            self.num_params += num_input * num_output + num_output

        self.population = []
        for i in range(population_size):
            self.population.append(np.random.randn(self.num_params))

        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.tournament_size = tournament_size

    def mutate(self, individual):
        mask = np.random.rand(len(individual)) < self.mutation_rate
        individual[mask] = np.random.randn(np.sum(mask)) * self.mutation_scale
        return individual

    def tournament_selection(self, fitness_scores):
        tournament_indices = np.random.choice(
            len(self.population),
            size=self.tournament_size,
            replace=False
        )

        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]

        return self.population[winner_idx].copy()

    def train(self, num_iterations):
        best_fitness_history = []
        best_individual = self.population[0]
        for iteration in range(num_iterations):
            fitness_scores = [fitness_function(ind) for ind in self.population]
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            best_individual = self.population[np.argmax(fitness_scores)]

            new_population = [best_individual.copy()]

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(fitness_scores)
                parent2 = self.tournament_selection(fitness_scores)

                child = crossover(parent1, parent2)
                child = self.mutate(child)

                new_population.append(child)
            self.population = new_population
        return best_individual, best_fitness_history


def fitness_function(individual, env, episodes=5):
    total_reward = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = env.action_space.sample() # fix later. need to place action from network

            state, reward, done, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / episodes


def crossover(parent1, parent2):
    mask = np.random.rand(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child


