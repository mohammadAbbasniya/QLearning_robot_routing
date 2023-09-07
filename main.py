import numpy as np
from random import Random

# --------------------- Definitions ---------------------
maze = np.array([
    # maze[i][j] != 0  -->  obstacle
    # maze[i][j] == 0  -->  open-square
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

rewards = np.array([
    # rewards[i][j] != 0  -->  non-terminal state
    # rewards[i][j] == 0  -->  terminal state
    [-50, -50, -50, -50, -50, -50, -50],
    [-50, 000, 000, 000, 000, 000, -50],
    [-50, 000, 000, 000, 000, 000, -50],
    [-50, 000, 000, 100, 000, 000, -50],
    [-50, 000, 000, 000, 000, 000, -50],
    [-50, -50, -50, -50, -50, -50, -50],
])

rows, cols = maze.shape
goal = 3, 3  # goal's position
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3  # actions index
N = {}  # access frequency table
Q = {}  # Q table
available_states = []  # list of non-terminal positions


# --------------------- Helper functions ---------------------
def is_inside(state):
    return 0 <= state[0] < rows and 0 <= state[1] < cols


def is_terminal(state):
    return rewards[state[0]][state[1]] != 0


def is_block(state):
    return maze[state[0]][state[1]] == 1


def init_tables():
    for i in range(rows):
        for j in range(cols):
            state = (i, j)
            if not is_block(state):
                available_states.append(state)
                for action in [UP, DOWN, LEFT, RIGHT]:
                    Q[state, action] = 0
                    N[state, action] = 0


def get_next_state(state, action):
    i, j = state
    if action == UP:
        return i - 1, j
    elif action == DOWN:
        return i + 1, j
    elif action == LEFT:
        return i, j - 1
    elif action == RIGHT:
        return i, j + 1


def get_transition_reward(state, next_state, action):
    reward = 0
    if is_terminal(next_state):
        reward = rewards[next_state[0]][next_state[1]]

    direction_costs = [3, 1, 2, 2]
    return reward - direction_costs[action]


def get_available_actions(state):
    actions = []
    for action in [UP, DOWN, LEFT, RIGHT]:
        if is_inside(get_next_state(state, action)):
            actions.append(action)

    return actions


def q_learning(trials, max_steps, epsilon, discount_factor, rnd):
    for trial in range(trials):
        # choose a random action
        state = rnd.choice(available_states)

        for step in range(max_steps):
            # check if we reached the goal
            if state == goal:
                break

            # choose an action
            available_actions = get_available_actions(state)
            r = rnd.uniform(0, 1)
            if r > epsilon:  # latest optimal action
                action = max(available_actions, key=lambda act: Q[state, act])
            else:  # random action
                action = rnd.choice(available_actions)

            # do the action
            next_state = get_next_state(state, action)
            if is_block(next_state):
                next_state = state  # bouncing back

            # update N(s, a) and Q(s, a)
            N[state, action] += 1

            reward = get_transition_reward(state, next_state, action)
            learning_rate = 1 / N[state, action]
            max_q = max(Q[next_state, UP], Q[next_state, DOWN], Q[next_state, LEFT], Q[next_state, RIGHT])
            Q[state, action] += learning_rate * (reward + (discount_factor * max_q) - Q[state, action])

            # go to the next state
            state = next_state

            # if state == goal:
            #     break


def table_to_string(T, precision):
    def D(x):
        return ('{' + f':.{precision}f' + '}').format(x)

    eol = '    '
    res = ''
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            if is_terminal(s) or is_block(s):
                res += "             " + eol
            else:
                res += f"{D(T[s, UP]):^13}" + eol
        res += '\n'

        for j in range(cols):
            s = (i, j)
            if is_terminal(s):
                res += f"{rewards[i][j]:^13}" + eol
            elif is_block(s):
                res += "     ####    " + eol
            else:
                res += f"{D(T[s, LEFT]):<6} {D(T[s, RIGHT]):>6}" + eol
        res += '\n'

        for j in range(cols):
            s = (i, j)
            if is_terminal(s) or is_block(s):
                res += "             " + eol
            else:
                res += f"{D(T[s, DOWN]):^13}" + eol
        res += '\n \n'

    return res


def max_in_directions(T):
    res = ''
    eol = '   '
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            if is_block(s):
                res += "####" + eol
            elif is_terminal(s):
                res += f"{rewards[i][j]:^4}" + eol
            else:
                max_q = np.argmax([T[s, UP], T[s, DOWN], T[s, LEFT], T[s, RIGHT]])
                res += ["^^^^", "vvvv", "<<<<", ">>>>"][max_q] + eol
        res += '\n'

    return res


# --------------------- Driver code ---------------------
if __name__ == '__main__':
    init_tables()
    q_learning(trials=50000,
               max_steps=100,
               epsilon=0.1,
               discount_factor=0.9,
               rnd=Random(32))

    print(table_to_string(N, precision=0), end='\n\n\n')
    print(table_to_string(Q, precision=1), end='\n\n\n')
    print(max_in_directions(Q))
