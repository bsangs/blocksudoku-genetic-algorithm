# ga.py
import threading
import random
import json
import os
import time
from game import Game
from score import ScoreManager
import tensorflow as tf
import numpy as np

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s) detected.")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. The code will run on CPU.")

class Individual:
    def __init__(self, model=None):
        if model is None:
            self.model = self.create_model()
            self.randomize_weights()
        else:
            self.model = self.create_model()
            self.set_weights(model.get_weights())
        self.fitness = 0

    def create_model(self):
        with tf.device('/GPU:0'):  # Explicitly assign the model to GPU:0
            inputs = tf.keras.Input(shape=(64 + 30,))
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(3, activation='linear')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def randomize_weights(self):
        for layer in self.model.layers:
            if hasattr(layer, 'kernel') and hasattr(layer, 'kernel_initializer'):
                new_kernel = layer.kernel_initializer(tf.TensorShape(layer.kernel.shape))
                layer.kernel.assign(new_kernel)
            if hasattr(layer, 'bias') and hasattr(layer, 'bias_initializer'):
                new_bias = layer.bias_initializer(tf.TensorShape(layer.bias.shape))
                layer.bias.assign(new_bias)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights_flat(self):
        weights = self.model.get_weights()
        flat = np.concatenate([w.flatten() for w in weights])
        return flat

    def set_weights_flat(self, flat_weights):
        shapes = [w.shape for w in self.model.get_weights()]
        new_weights = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            new_weights.append(flat_weights[idx:idx+size].reshape(shape))
            idx += size
        self.model.set_weights(new_weights)

    def decide_action(self, game):
        """
        Decide action based on the current game state using the neural network model.
        Returns: (block_index, x, y) or None if no valid moves.
        """
        with game.lock:
            current_blocks = game.get_current_blocks()
            if len(current_blocks) == 0:
                return None  # No blocks to place

            # Extract game state features
            game_state = self.extract_game_state(game)  # Implement this method

        # Convert game state to TensorFlow tensor and perform a forward pass
        game_state_tensor = tf.convert_to_tensor([game_state], dtype=tf.float32)
        predictions = self.model(game_state_tensor, training=False)
        
        # Interpret predictions to decide on an action
        action_index = tf.argmax(predictions, axis=1).numpy()[0]
        
        # Convert action_index to (block_index, x, y)
        action = self.map_action_index(action_index, game)  # Implement this method
        
        return action

    def extract_game_state(self, game):
        """
        게임의 현재 상태를 추출하여 특징 벡터로 반환합니다.
        실제 게임 로직에 맞게 구현해야 합니다.
        """
        # 예시: 임의의 94차원 벡터 반환 (64 + 30)
        # 실제로는 게임의 상태를 반영한 특징을 추출해야 합니다.
        return np.random.rand(64 + 30)

    def map_action_index(self, action_index, game):
        """
        모델의 출력 인덱스를 실제 게임 액션으로 매핑합니다.
        실제 게임 로직에 맞게 구현해야 합니다.
        """
        # 예시: 가능한 유효한 움직임을 랜덤으로 선택
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return valid_moves[action_index % len(valid_moves)]

class GeneticAlgorithm:
    def __init__(self, pop_size=20, generations=1000, games=None, gui=None, mutation_rate=0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.population = [Individual() for _ in range(pop_size)]
        self.games = games if games else [Game() for _ in range(pop_size)]
        self.gui = gui
        self.mutation_rate = mutation_rate
        self.current_generation = 0
        self.best_score = 0
        self.generations_without_improvement = 0
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.save_interval = 100  # Save every 100 generations

        # Statistics
        self.average_score = 0
        self.min_score = 0

    def evaluate_individual(self, individual, game):
        with game.lock:
            game.reset()
            game.strategy = individual  # Assign the strategy

        # Simulate the game step by step, allowing GPU operations
        while not game.game_over:
            placed, removed = game.make_move_ai()
            # Optionally, include a small sleep if necessary
            # time.sleep(0.01)

        with game.lock:
            fitness = game.get_score()

        with self.lock:
            individual.fitness = fitness
            if fitness > self.best_score:
                self.best_score = fitness
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

    def get_game_state(self, individual):
        """
        개체의 게임 상태를 추출하여 반환합니다.
        실제 게임 로직에 맞게 구현해야 합니다.
        """
        game = self.games[self.population.index(individual)]
        # 게임에서 필요한 특징을 추출하여 반환
        # 예시: 임의의 94차원 벡터 반환 (64 + 30)
        # 실제로는 게임의 상태를 반영한 특징을 추출해야 합니다.
        return np.random.rand(64 + 30)

    def simulate_game(self, individual):
        """
        개체의 전략을 사용하여 게임을 시뮬레이션하고 피트니스 점수를 반환합니다.
        실제 게임 로직에 맞게 구현해야 합니다.
        """
        game = self.games[self.population.index(individual)]
        with game.lock:
            game.reset()
            game.strategy = individual

        while not game.game_over:
            placed, removed = game.make_move_ai()
            # Optionally, include a small sleep if necessary
            # time.sleep(0.01)

        with game.lock:
            return game.get_score()

    def evaluate_population(self):
        inputs = []
        for individual in self.population:
            # 실제 게임 상태에 맞게 게임 상태 특징을 추출
            game_state = self.get_game_state(individual)
            inputs.append(game_state)
        
        # Convert inputs to a TensorFlow tensor
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        
        # Perform a batch forward pass on the GPU
        # 여기서는 각 개체의 모델을 개별적으로 실행하지만, 더 최적화할 수 있습니다.
        predictions = []
        for i, individual in enumerate(self.population):
            prediction = individual.model(inputs_tensor[i:i+1])[0]
            predictions.append(prediction)
        predictions = tf.stack(predictions)
        
        # Assign fitness based on predictions; define how predictions map to fitness
        # 여기서는 단순히 게임을 시뮬레이션하여 피트니스를 할당합니다.
        for individual in self.population:
            individual.fitness = self.simulate_game(individual)  # Implement this method based on your Game class
        
        # Compute statistics
        with self.lock:
            scores = [ind.fitness for ind in self.population]
            self.best_score = max(scores)
            self.min_score = min(scores)
            self.average_score = sum(scores) / len(scores)

    def select_parents(self):
        # Tournament selection example
        tournament_size = 5
        parents = []
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            parent = max(tournament, key=lambda ind: ind.fitness)
            parents.append(parent)
        return parents

    def crossover(self, parent1, parent2):
        child = Individual()
        weights1 = parent1.get_weights_flat()
        weights2 = parent2.get_weights_flat()
        mask = np.random.rand(len(weights1)) > 0.5
        child_weights = np.where(mask, weights1, weights2)
        child.set_weights_flat(child_weights)
        return child

    def mutate(self, individual):
        # Convert weights to TensorFlow tensor for GPU operations
        weights = tf.convert_to_tensor(individual.get_weights_flat(), dtype=tf.float32)
        
        # Create a mutation mask using TensorFlow operations
        mutation_mask = tf.random.uniform(shape=weights.shape) < self.mutation_rate
        
        # Generate Gaussian noise for mutation
        noise = tf.random.normal(shape=weights.shape, mean=0.0, stddev=0.1)
        
        # Apply mutations where the mask is True
        mutated_weights = tf.where(mutation_mask, weights + noise, weights)
        
        # Update the individual's weights
        individual.set_weights_flat(mutated_weights.numpy())

    def create_next_generation(self):
        new_population = []
        
        # Elitism: retain the top 2 individuals
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        elite = sorted_pop[:2]
        new_population.extend(elite)
        
        # Fill the rest of the population
        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            if random.random() < 0.8:  # Crossover rate 80%
                child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        with self.lock:
            self.population = new_population[:self.pop_size]

    def save_population(self):
        # Collect data within the lock
        with self.lock:
            data = {
                "generation": self.current_generation,
                "best_score": self.best_score,
                "average_score": self.average_score,
                "min_score": self.min_score,
                "population": [ind.get_weights_flat().tolist() for ind in self.population]
            }
        # Save to file
        try:
            with open("population.json", "w") as f:
                json.dump(data, f)
            print("Population saved to population.json")
        except Exception as e:
            print(f"Error saving population: {e}")

    def load_population(self):
        if not os.path.exists("population.json"):
            print("No saved population found.")
            return
        with self.lock:
            with open("population.json", "r") as f:
                data = json.load(f)
            self.current_generation = data["generation"]
            self.best_score = data["best_score"]
            self.average_score = data["average_score"]
            self.min_score = data["min_score"]
            for ind, weights in zip(self.population, data["population"]):
                ind.set_weights_flat(np.array(weights))
        print("Population loaded from population.json")

    def run(self):
        while self.current_generation < self.generations and not self.stop_event.is_set():
            # Evaluate the entire population
            self.evaluate_population()
            
            # Create the next generation
            self.create_next_generation()
            
            with self.lock:
                self.current_generation += 1
                if self.current_generation % self.save_interval == 0:
                    self.save_population()
            # Optionally, include a small sleep if necessary
            # time.sleep(0.01)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.stop_event.clear()
        print("Genetic Algorithm stopped.")
