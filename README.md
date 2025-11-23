# Q Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
→ Initialize Q-table and hyperparameters.<br>
→ Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.<br>
→ After training, derive the optimal policy from the Q-table.<br>
→ Implement the Monte Carlo method to estimate state values.<br>
→ Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.<br>

## Q LEARNING FUNCTION
#### Name: ROHIT G
#### Register Number: 212222240083
```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
          action = select_action(state, Q, epsilon[e])
          next_state, reward, done, _ = env.step(action)
          td_target = reward + gamma * Q[next_state].max() * (not done)
          td_error = td_target - Q[state][action]
          Q[state][action] = Q[state][action] + alphas[e] * td_error
          state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```





## OUTPUT:
### Optimal State Value Functions:
<img width="963" height="48" alt="image" src="https://github.com/user-attachments/assets/5afec2f0-bd2e-4060-afad-8a2e7d7ae288" />


### Optimal Action Value Functions:
<img width="1156" height="656" alt="image" src="https://github.com/user-attachments/assets/eff35610-b116-48b2-ab4c-0692008e3d54" />


### State value functions of Monte Carlo method:
<img width="1623" height="735" alt="image" src="https://github.com/user-attachments/assets/eb240b22-de0f-4015-b29e-6987bd42c03f" />


### State value functions of Qlearning method:
<img width="1635" height="735" alt="image" src="https://github.com/user-attachments/assets/8a4af37f-00b9-4fcd-9f3d-51f99462a397" />


## RESULT:
Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
