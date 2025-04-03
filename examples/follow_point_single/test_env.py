from set_environment import Environment

# test_env.py


def test_environment():
    import numpy as np

    env = Environment(n_elem=50, mode=1, target_position=np.array([0.05, 0.05, 0.05]))
    #state = env.reset() #initializes with params

    num_steps = 100

    for step in range(num_steps):
        action = np.random.uniform(0, env.max_tension, size=(4,))  # Random action
        state, reward, done, _ = env.step(action)  # Take a step in the environment
        
        print(f"Step {step}: Reward = {reward}, State = {state}")
        
        if done:
            print("Episode finished")
            break

if __name__ == "__main__":
    test_environment()
    print("tested")