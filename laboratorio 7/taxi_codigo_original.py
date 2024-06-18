#Integrantes:  Pinto Mamani Paola Andrea Ing. de Sistemas
#              Zanabria Vega Maria Alejandra Ing de Sistemas
#Github: https://github.com/MAZanabria/InteligenciaArtificialZVMA/tree/pacman/Laboratorio%207

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    env = gym.make('Taxi-v3')
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo Q-learning
    learning_rate = 0.01  # Tasa de aprendizaje
    discount_factor = 0.95  # Factor de descuento de la recompensa
    epsilon = 1.0
    epsilon_decay_rate =  0.005
    rng = np.random.default_rng()

    # Inicializa un array para almacenar las recompensas obtenidas
    rewards_per_episode = np.zeros(episodes)  # Recompensa por episodio

    # Bucle principal de entrenamiento
    for i in range(episodes):
        if (i + 1) % 1000 == 0:
            env.close()
            env = gym.make('Taxi-v3', render_mode='human')
        else:
            env.close()
            env = gym.make('Taxi-v3')

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]  # Obtén el primer elemento de la tupla

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Decisión de acción: explorar o explotar
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = np.argmax(q_table[state,:])

            # Realiza la acción y obtiene el nuevo estado, la recompensa y los indicadores de terminación y truncado
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza la tabla Q con la nueva información
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action])

            # Actualiza el estado para el siguiente paso
            state = new_state  # Obtén el primer elemento de la nueva tupla de estado

            # Reduce epsilon para disminuir la exploración a lo largo del tiempo
            epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Registra si el agente obtuvo una recompensa en este episodio
        if reward == 20:  # El reward para llegar al objetivo en Taxi-v3 es 20
            rewards_per_episode[i] = 1

        # Imprime el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
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
