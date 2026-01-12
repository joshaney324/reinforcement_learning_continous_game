import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from pong import Environment
from models.genetic_algorithm import GeneticAlgorithm, get_action, crossover, fitness_function


def train_generation(ga, env, architecture):
    """Train a single generation and return best individual and fitness scores."""
    fitness_scores = []

    for individual in ga.population:
        fitness = fitness_function(individual, architecture, ga.population, env)
        fitness_scores.append(fitness)

    best_fitness = max(fitness_scores)
    best_idx = np.argmax(fitness_scores)
    best_individual = ga.population[best_idx].copy()

    # Create new population with elitism
    new_population = [best_individual.copy()]

    while len(new_population) < ga.population_size:
        parent1 = ga.tournament_selection(fitness_scores)
        parent2 = ga.tournament_selection(fitness_scores)
        child = crossover(parent1, parent2)
        child = ga.mutate(child)
        new_population.append(child)

    ga.population = new_population

    return best_individual, best_fitness, np.mean(fitness_scores)


def play_against_agent(individual, architecture, num_games=3):
    """Play against the trained agent - you control the right paddle."""
    viz_env = Environment(render=True)

    wins_ai = 0
    wins_player = 0

    print("\n" + "=" * 50)
    print("CONTROLS:")
    print("W or UP ARROW    - Move paddle up")
    print("S or DOWN ARROW  - Move paddle down")
    print("ESC or Q         - Quit")
    print("=" * 50)

    for game in range(num_games):
        state = viz_env.reset_game()
        done = False
        steps = 0
        max_steps = 5000

        print(f"\nGame {game + 1}/{num_games}")

        while not done and steps < max_steps:
            action_player = 0  # Default: no movement

            # Handle player input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return wins_ai, wins_player
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        pygame.quit()
                        return wins_ai, wins_player

            # Check continuous key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                action_player = -1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                action_player = 1

            # AI controls left paddle
            action_ai = get_action(architecture, individual, state) - 1

            state, reward_left, reward_right, done = viz_env.step(action_ai, action_player)
            steps += 1

        # Track wins
        if viz_env.score_left > viz_env.score_right:
            wins_ai += 1
            print(f"AI WINS! Final Score - AI: {viz_env.score_left}, You: {viz_env.score_right}")
        else:
            wins_player += 1
            print(f"YOU WIN! Final Score - AI: {viz_env.score_left}, You: {viz_env.score_right}")

    print(f"\n" + "=" * 50)
    print(f"FINAL RESULTS: AI: {wins_ai} wins, You: {wins_player} wins")
    print("=" * 50)

    pygame.quit()
    return wins_ai, wins_player


def main():
    # Network architecture: 6 inputs (state), 12 hidden neurons, 3 outputs (up, stay, down)
    network_architecture = [6, 12, 3]

    # GA hyperparameters
    population_size = 50
    mutation_rate = 0.1
    mutation_scale = 0.3
    tournament_size = 5
    num_generations = 200

    print("=" * 50)
    print("Pong Genetic Algorithm Training")
    print("=" * 50)
    print(f"Network Architecture: {network_architecture}")
    print(f"Population Size: {population_size}")
    print(f"Generations: {num_generations}")
    print(f"Mutation Rate: {mutation_rate}")
    print("=" * 50)

    # Initialize
    ga = GeneticAlgorithm(
        network_architecture=network_architecture,
        population_size=population_size,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        tournament_size=tournament_size
    )

    env = Environment(render=False)

    best_fitness_history = []
    avg_fitness_history = []
    best_overall = None
    best_overall_fitness = float('-inf')

    print("\nTraining...")

    for generation in range(num_generations):
        best_individual, best_fitness, avg_fitness = train_generation(ga, env, network_architecture)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall = best_individual.copy()

        if (generation + 1) % 5 == 0 or generation == 0:
            print(
                f"Gen {generation + 1:3d} | Best: {best_fitness:8.2f} | Avg: {avg_fitness:8.2f} | Overall Best: {best_overall_fitness:8.2f}")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Fitness Achieved: {best_overall_fitness:.2f}")
    print("=" * 50)

    # Save the best weights
    np.save('best_pong_agent.npy', best_overall)
    print("Best agent saved to 'best_pong_agent.npy'")

    # Play against the trained agent
    print("\nGet ready to play against the AI!")
    input("Press ENTER to start...")
    play_against_agent(best_overall, network_architecture, num_games=5)


if __name__ == "__main__":
    main()