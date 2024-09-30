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
        inputs = tf.keras.Input(shape=(64 + 30,))  # type: ignore
        x = tf.keras.layers.Dense(64, activation='relu')(inputs) # type: ignore    
        x = tf.keras.layers.Dense(128, activation='relu')(x) # type: ignore
        x = tf.keras.layers.Dense(128, activation='relu')(x) # type: ignore
        x = tf.keras.layers.Dense(64, activation='relu')(x) # type: ignore
        outputs = tf.keras.layers.Dense(3, activation='linear')(x) # type: ignore
        model = tf.keras.Model(inputs=inputs, outputs=outputs) # type: ignore
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
        현재 게임 상태를 기반으로 행동을 결정합니다.
        반환: (block_index, x, y) 또는 유효한 움직임이 없으면 None.
        """
        with game.lock:
            current_blocks = game.get_current_blocks()
            if len(current_blocks) == 0:
                return None  # 놓을 블록이 없음

            # 가능한 모든 유효한 움직임 생성
            valid_moves = []
            for block_index, block in enumerate(current_blocks):
                for x in range(game.board_size):
                    for y in range(game.board_size):
                        if game.can_place(block, x, y):
                            valid_moves.append((block_index, x, y))

            if not valid_moves:
                return None  # 유효한 움직임이 없음

        # 현재는 랜덤으로 유효한 움직임 선택
        return random.choice(valid_moves)


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

        # Simulate the game step by step, allowing GUI to update
        while not game.game_over:
            placed, removed = game.make_move_ai()
            # Sleep to allow GUI to reflect updates
            # time.sleep(0.01)  # Adjust as needed

        with game.lock:
            fitness = game.get_score()

        with self.lock:
            individual.fitness = fitness
            if fitness > self.best_score:
                self.best_score = fitness
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

    def evaluate_population(self):
        threads = []
        for individual, game in zip(self.population, self.games):
            thread = threading.Thread(target=self.evaluate_individual, args=(individual, game))
            thread.start()
            threads.append(thread)
        # 모든 스레드가 완료될 때까지 대기
        for thread in threads:
            thread.join()

        # 모든 평가가 끝난 후 통계 계산
        with self.lock:
            scores = [ind.fitness for ind in self.population]
            self.best_score = max(scores)
            self.min_score = min(scores)
            self.average_score = sum(scores) / len(scores)


    def select_parents(self):
        # 토너먼트 선택 예시
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
        weights = individual.get_weights_flat()
        mutation_mask = np.random.rand(len(weights)) < self.mutation_rate
        # Apply mutation by adding Gaussian noise
        weights[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask)) # type: ignore
        individual.set_weights_flat(weights)

    def create_next_generation(self):
        new_population = []
        
        # 엘리트주의: 상위 2명 유지
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        elite = sorted_pop[:2]
        new_population.extend(elite)
        
        # 나머지 인구 채우기
        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            if random.random() < 0.8:  # 교차 비율 80%
                child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        with self.lock:
            self.population = new_population[:self.pop_size]

    def save_population(self):
        # 잠금 내에서 데이터 수집
        with self.lock:
            data = {
                "generation": self.current_generation,
                "best_score": self.best_score,
                "average_score": self.average_score,
                "min_score": self.min_score,
                "population": [ind.get_weights_flat().tolist() for ind in self.population]
            }
        # 잠금 해제 후 파일 저장
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
            # print(f"Generation {self.current_generation}")
            self.evaluate_population()
            self.create_next_generation()
            with self.lock:
                self.current_generation += 1
                if self.current_generation % self.save_interval == 0:
                    self.save_population()
            # Sleep to allow GUI to update
            # time.sleep(0.01)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.stop_event.clear()
        print("Genetic Algorithm stopped.")
