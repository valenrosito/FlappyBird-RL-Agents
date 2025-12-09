# Conclusiones Trabajo Practico

## 1. Ingeniería de Características y Discretización del Estado

### Variables de Estado Seleccionadas

Se identificaron **4 variables clave** que capturan la información esencial para tomar decisiones en el juego:

| Variable | Descripción |
|----------|-------------|
| `player_y` | Altura vertical del pájaro |
| `player_vel_y` | Velocidad vertical del pájaro |
| `next_pipe_dist_x` | Distancia horizontal al próximo tubo |
| `next_pipe_dy_to_mid` | Distancia vertical al centro del hueco |

### Proceso de Discretización

Se establecieron los siguientes rangos basados en el análisis estadístico del juego:

```python
RANGES = {
    'player_y': (0, 512),
    'player_vel_y': (-15, 15),
    'next_pipe_dist_x': (0, 288),
    'next_pipe_dy_to_mid': (-200, 200)
}
```

Se utilizaron **20 bins por dimensión**, resultando en un espacio de **160,000 estados discretos** ($20^4$). Esta discretización balancea precisión y eficiencia, permite usar Q-Learning tradicional con convergencia rápida y alta interpretabilidad, aunque implica pérdida de información continua.

---

## 2. Análisis y Comparación de Resultados

### 2.1 Q-Learning Agent (Discreto)

**Rendimiento:** >300 puntos | Precisión "casi quirúrgica" | Convergencia exitosa

**Características:**
- Q-table con 160,000 estados discretizados
- Consultas instantáneas sin overhead computacional
- Estrategia ε-greedy

**Ventajas:** Decisiones rápidas y deterministas, alta interpretabilidad.  
**Desventajas:** No escala a problemas complejos, dependiente de la discretización.  

### 2.2 Deep Q-Network Agent (DQN)

**Rendimiento:** ~20 puntos | Dificultades con subidas repentinas | Efecto "cámara lenta"

**Características:**
- Red neuronal como aproximador Q(s,a)
- Replay buffer para aprendizaje
- Inicializado con Q-table del agente discreto

**Ventajas:** Escalable teóricamente, generaliza patrones.  
**Desventajas:** Latencia crítica en inferencia, rendimiento muy inferior (20 vs >300 puntos).  

---

### 2.3 Comparación Directa

| Aspecto | Q-Learning (Discreto) | Deep Q-Network |
|---------|----------------------|----------------|
| **Puntuación máxima** | >300  | ~20 |
| **Velocidad de ejecución** | Instantánea  | Lenta (overhead de NN) |
| **Convergencia** | Rápida y estable | Más lenta e inestable |
| **Precisión de acciones** | Alta (quirúrgica) | Moderada-Baja |
| **Escalabilidad** | Limitada | Alta (teórica) |
| **Complejidad implementación** | Baja | Alta |
| **Interpretabilidad** | Alta | Baja (caja negra) |
| **Requisitos computacionales** | Mínimos | Altos |

---

### 2.4 Análisis de las Diferencias de Rendimiento

#### ¿Por qué el DQN tuvo peor desempeño?

1. **Latencia en la inferencia**: 
   - El costo computacional de propagar hacia adelante en la red neuronal introduce un delay
   - En un juego de reacciones rápidas como Flappy Bird, este delay es crítico
   - El efecto "cámara lenta" observado evidencia este problema

2. **Problema de la dimensionalidad justa**:
   - Flappy Bird tiene un espacio de estados relativamente pequeño (4 variables)
   - Para este tipo de problemas, Q-Learning tabular es más eficiente
   - Las redes neuronales son más apropiadas para problemas con alta dimensionalidad (imágenes, muchos sensores)

3. **Dificultad en subidas repentinas**:
   - Las redes neuronales generalizan suavemente
   - Pueden tener dificultad para aprender políticas con cambios abruptos
   - El Q-Learning discreto puede memorizar acciones específicas para cada estado

4. **Transfer learning incompleto**:
   - Aunque se inicializó con la Q-table del agente discreto, la representación puede no ser óptima
   - La red debe aprender a representar la función Q en un espacio continuo, lo cual es un desafío adicional

---

## Conclusión

En este proyecto, **Q-Learning con discretización demostró ser la solución óptima** para Flappy Bird, logrando rendimiento superior con menor complejidad.
