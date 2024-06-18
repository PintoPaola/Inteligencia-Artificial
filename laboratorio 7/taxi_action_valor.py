#Integrantes:  Pinto Mamani Paola Andrea Ing. de Sistemas
#              Zanabria Vega Maria Alejandra Ing de Sistemas
#Github: https://github.com/MAZanabria/InteligenciaArtificialZVMA/tree/pacman/Laboratorio%207
#Github: https://github.com/PintoPaola/Inteligencia-Artificial/tree/main/laboratorio%207
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    # Crea el entorno del juego Taxi-v3
    env = gym.make('Taxi-v3')
    
    # Inicializa la tabla Q con ceros: dimensiones son (número de estados, número de acciones)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo Q-learning
    learning_rate = 0.01  # Tasa de aprendizaje
    discount_factor = 0.95  # Factor de descuento de la recompensa
    epsilon = 1.0  # Factor de exploración inicial
    epsilon_decay_rate = 0.00005  # Tasa de decaimiento de epsilon
    rng = np.random.default_rng()  # Generador de números aleatorios
    
    # Array para almacenar las recompensas obtenidas por episodio
    rewards_per_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento a través de episodios
    for i in range(episodes):
        # Cada 1000 episodios se muestra el entorno en modo humano para visualización
        if (i + 1) % 1000 == 0:
            env.close()  # Cierra el entorno actual
            env = gym.make('Taxi-v3', render_mode='human')  # Crea un nuevo entorno con modo renderizado humano
        else:
            env.close()  # Cierra el entorno actual
            env = gym.make('Taxi-v3')  # Crea un nuevo entorno sin renderizado humano

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]  # Obtén el primer elemento de la tupla del estado inicial

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Decisión de acción: explorar o explotar
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración: elige una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: elige la mejor acción según la tabla Q

            # Realiza la acción y obtiene el nuevo estado, la recompensa y los indicadores de terminación y truncado
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza la tabla Q con la nueva información usando Q-learning
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action])

            # Actualiza el estado para el siguiente paso
            state = new_state  # Actualiza el estado actual al nuevo estado

            # Reduce epsilon para disminuir la exploración a lo largo del tiempo, asegurándose de que no caiga por debajo de 0.01
            epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Registra la recompensa obtenida en este episodio
        rewards_per_episode[i] = reward

        # Imprime un resumen cada 50 episodios
        if (i + 1) % 50 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla Q final para inspección
    print("Mejor Q:")
    print(q_table)

    # Grafica las recompensas acumuladas
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas acumuladas (últimos 100 episodios)')
    plt.title('Recompensas durante el entrenamiento')
    plt.show()

# Ejecuta la función de entrenamiento si el script es el programa principal
if __name__ == '__main__':
    train(15000)

#el método de acción-valor se aplica a través de la tabla Q en el algoritmo Q-learning, 
 #donde el agente aprende a seleccionar acciones óptimas para maximizar las recompensas esperadas en el entorno del juego "Taxi-v3".
 #El uso de 𝜖 ϵ-greedy asegura que el agente explore y explote de manera balanceada para aprender una
 #política de acción efectiva a lo largo del entrenamiento.