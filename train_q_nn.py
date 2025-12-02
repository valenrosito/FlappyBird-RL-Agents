import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table_final.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Normalizar X: los bins van de 0-19, normalizar a [0, 1]
X = X / 19.0

print(f"Rango de X después de normalización: [{X.min():.2f}, {X.max():.2f}]")
print(f"Rango de y (Q-values): [{y.min():.2f}, {y.max():.2f}]")

# --- Definir la red neuronal ---
model = keras.Sequential([
    layers.Input(shape=(4,)),  # 4 características normalizadas
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),  
    layers.Dropout(0.25),
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(2, activation='linear')
])

# Learning rate más bajo desde el inicio
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
              loss='mse',
              metrics=['mae'])

print("\n=== Resumen del Modelo de Red Neuronal ===")
model.summary()
print("--------------------------------------------")
print(f"\nNúmero de estados en Q-table: {len(X)}")
print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
print("--------------------------------------------")

# --- Entrenar la red neuronal ---
print(f"\nIniciando entrenamiento...")
history = model.fit(
    X, y,
    epochs=500,              
    batch_size=64,           
    validation_split=0.2,    
    verbose=1,
    shuffle=True,            # Mezclar datos cada época               
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ), 
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            verbose=1
        )
    ]
)

# --- Mostrar resultados del entrenamiento ---
print("\n=== Resultados del Entrenamiento ===")
print(f"Loss final (entrenamiento): {history.history['loss'][-1]:.4f}")
print(f"Loss final (validación): {history.history['val_loss'][-1]:.4f}")
print(f"MAE final (entrenamiento): {history.history['mae'][-1]:.4f}")
print(f"MAE final (validación): {history.history['val_mae'][-1]:.4f}")
print(f"Épocas entrenadas: {len(history.history['loss'])}")

# --- Guardar el modelo entrenado ---
model.save('flappy_q_nn_model.keras')
print('\n✓ Modelo guardado como flappy_q_nn_model.keras')

