class Agent:
    """Clase base para agentes."""
    def __init__(self, actions, game=None):
        self.actions = actions
        self.game = game
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

    def reset(self):
        """Reinicia el estado interno del agente (si es necesario)."""
        pass

    def act(self, state):
        """Devuelve la acción a tomar dado el estado actual."""
        raise NotImplementedError
