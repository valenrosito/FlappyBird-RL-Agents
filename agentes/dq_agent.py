from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    Nota: epsilon se define 0.0 por defecto para que el agente no tome acciones aleatorias en test.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
            epsilon=0.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table_final.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        

    def act(self, state):
        """
        Elige una acción usando epsilon-greedy sobre la Q-table.
        COMPLETAR: Implementar la política epsilon-greedy.
        """
        discrete_state = self.discretize_state(state)

        # 2. Elegir acción: Epsilon-Greedy
        if random.random() < self.epsilon:
            # Exploración: Elegir acción aleatoria
            # self.actions es una lista de las posibles acciones (ej. [0, 1] o ['NO_FLAP', 'FLAP'])
            action = random.choice(self.actions)
        else:
            # Explotación: Elegir la mejor acción (máximo Q-value)
            
            # Obtener los Q-values para el estado discreto. 
            # defaultdict garantiza que devuelve np.zeros(len(self.actions)) si el estado es nuevo.
            q_values = self.q_table[discrete_state]
            
            # Encontrar el índice de la acción con el Q-value máximo
            best_action_index = np.argmax(q_values)
            
            # Devolver la acción correspondiente a ese índice
            action = self.actions[best_action_index]
            
        return action

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
