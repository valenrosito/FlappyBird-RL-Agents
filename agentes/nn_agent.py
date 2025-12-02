from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)
    

    def act(self, state):
        """
        Usa la red neuronal para predecir Q-values y elegir la mejor acción.
        """
        # 1. Discretizar el estado (igual que en el Q-Agent)
        discrete_state = self.discretize_state(state)
        
        # 2. Convertir a numpy array y normalizar (IMPORTANTE: igual que en entrenamiento)
        state_array = np.array([discrete_state], dtype=np.float32)  # Shape: (1, 4)
        state_normalized = state_array / 19.0  # Normalizar bins [0-19] -> [0-1]
        
        # 3. Predecir Q-values con la red neuronal
        q_values = self.model.predict(state_normalized, verbose=0)[0]  # Shape: (2,)
        
        # 4. Elegir la acción con el mayor Q-value
        best_action_index = np.argmax(q_values)
        action = self.actions[best_action_index]
        
        return action
