from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import argparse
import importlib
import sys
import csv
from datetime import datetime
import os

# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = FlappyBird()  
env = PLE(game, display_screen=True, fps=30) # fps=30 es más normal, display_screen=True para ver


# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet()

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Test de agentes para FlappyBird (PLE)")
parser.add_argument('--agent', type=str, required=True, help='Ruta completa del agente, ej: agentes.random_agent.RandomAgent')
args = parser.parse_args()

# --- Carga dinámica del agente usando path completo ---
try:
    module_path, class_name = args.agent.rsplit('.', 1)
    agent_module = importlib.import_module(module_path)
    AgentClass = getattr(agent_module, class_name)
except (ValueError, ModuleNotFoundError, AttributeError):
    print(f"No se pudo encontrar la clase {args.agent}")
    sys.exit(1)

# Inicializar el agente
agent = AgentClass(actions, game)

# --- Crear archivo CSV para guardar estados ---
os.makedirs('game_data', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'game_data/states_{timestamp}.csv'

# Abrir archivo CSV y escribir encabezados
csv_file = open(csv_filename, 'w', newline='')
csv_writer = None  # Se inicializará con las claves del primer state_dict

print(f"Guardando estados en: {csv_filename}")

# Agente con acciones aleatorias
try:
    while True:
        env.reset_game()
        agent.reset()
        state_dict = env.getGameState()
        done = False
        total_reward_episode = 0
        print("\n--- Ejecutando agente ---")
        
        # Inicializar el CSV writer con las claves del primer estado
        if csv_writer is None:
            fieldnames = ['episode', 'step', 'reward', 'total_reward'] + list(state_dict.keys())
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
        
        episode_num = 0
        step = 0
        
        while not done:
            action = agent.act(state_dict)
            reward = env.act(action)
            state_dict = env.getGameState()
            done = env.game_over()
            total_reward_episode += reward
            
            # Guardar estado en CSV
            row = {
                'episode': episode_num,
                'step': step,
                'reward': reward,
                'total_reward': total_reward_episode,
                **state_dict  # Desempaquetar el diccionario de estado
            }
            csv_writer.writerow(row)
            csv_file.flush()  # Forzar escritura inmediata
            
            step += 1
            time.sleep(0.03)
        
        print(f"Recompensa episodio: {total_reward_episode+5}")
        episode_num += 1

except KeyboardInterrupt:
    print(f"\n\nJuego interrumpido. Estados guardados en: {csv_filename}")
    csv_file.close()
