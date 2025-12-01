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
            epsilon=0.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table.pkl"):
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
            
        self.N_BINS = 20
        self.RANGES = {
            'player_y': (0, 512),       # Altura del pájaro 
            'player_vel_y': (-15, 15),   # Velocidad vertical del pájaro
            'next_pipe_dist_x': (0, 288), # Distancia horizontal al tubo
            'next_pipe_dy_to_mid': (-200, 200) # Distancia vertical al centro del hueco
        }

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        COMPLETAR: Implementar la discretización adecuada para el entorno.
        """
        # Acceder a los valores del estado (cambiar si la estructura de 'state' es diferente)
        player_y = state['player_y']
        player_vel_y = state['player_vel'] 
        next_pipe_dist_x = state['next_pipe_dist_to_player'] 
        
        # Calcular la distancia vertical relativa al centro del hueco
        pipe_center_y = (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) / 2
        next_pipe_dy_to_mid = pipe_center_y - player_y # Distancia vertical al centro del hueco
        
        def get_bin(value, range_min, range_max, num_bins):
            """Calcula el índice del bin para un valor dado."""
            value = max(range_min, min(range_max, value))
            bin_size = (range_max - range_min) / num_bins
            bin_index = int((value - range_min) / bin_size)
            return min(bin_index, num_bins - 1)

        # Discretizar cada variable
        y_bin = get_bin(player_y, *self.RANGES['player_y'], self.N_BINS)
        vel_bin = get_bin(player_vel_y, *self.RANGES['player_vel_y'], self.N_BINS)
        dist_x_bin = get_bin(next_pipe_dist_x, *self.RANGES['next_pipe_dist_x'], self.N_BINS)
        dist_y_bin = get_bin(next_pipe_dy_to_mid, *self.RANGES['next_pipe_dy_to_mid'], self.N_BINS)
        
        # El estado discreto es la tupla de 4 dimensiones
        return (y_bin, vel_bin, dist_x_bin, dist_y_bin)

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
