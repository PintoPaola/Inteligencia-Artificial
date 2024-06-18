#Integrantes:  Pinto Mamani Paola Andrea Ing. de Sistemas
#              Zanabria Vega Maria Alejandra Ing de Sistemas
#Github: https://github.com/MAZanabria/InteligenciaArtificialZVMA/tree/pacman/Laboratorio%207
#Github: https://github.com/PintoPaola/Inteligencia-Artificial/tree/main/laboratorio%207
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes, alpha=0.01):
    env = gym.make('Taxi-v3')
    preference_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Parámetros del algoritmo de gradiente
    discount_factor = 0.95  # Factor de descuento de la recompensa
    epsilon = 1.0
    epsilon_decay_rate = 0.005
    rng = np.random.default_rng()

    # Inicializa un array para almacenar las recompensas obtenidas
    rewards_per_episode = np.zeros(episodes)  # Recompensa por episodio
    avg_reward = 0

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
            # Calcula la política (probabilidades de seleccionar cada acción)
            exp_preferences = np.exp(preference_table[state])
            policy = exp_preferences / np.sum(exp_preferences)

            # Decisión de acción: explorar o explotar
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = rng.choice(np.arange(env.action_space.n), p=policy)

            # Realiza la acción y obtiene el nuevo estado, la recompensa y los indicadores de terminación y truncado
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza el promedio de recompensa
            avg_reward = avg_reward + (reward - avg_reward) / (i + 1)

            # Actualiza las preferencias de las acciones utilizando el gradiente
            for a in range(env.action_space.n):
                if a == action:
                    preference_table[state, a] += alpha * (reward - avg_reward) * (1 - policy[a])
                else:
                    preference_table[state, a] -= alpha * (reward - avg_reward) * policy[a]

            # Actualiza el estado para el siguiente paso
            state = new_state  # Obtén el primer elemento de la nueva tupla de estado

            # Reduce epsilon para disminuir la exploración a lo largo del tiempo
            epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Registra la recompensa obtenida en este episodio
        rewards_per_episode[i] = reward

        # Imprime el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio: {i + 1} - Recompensa acumulada: {np.sum(rewards_per_episode[max(0, i - 99):i + 1])}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla de preferencias final para inspección
    print("Mejor tabla de preferencias:")
    print(preference_table)

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
