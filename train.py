import stable_baselines3 as sb3
from env import Env2048

# trains model over certain amount of time steps, logs after n time steps, does this N iterations
def train(model, log_name, timesteps=10000, iters=100):
    for i in range(1, iters + 1):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=log_name)
        model.save(f"models/{log_name}-{i}")

# for retraining an already trained model
def retrain(env, log_name, timesteps, iters):
    name, steps = log_name.split('-') # get name of model, time steps
    model = sb3.PPO.load(f'models/{log_name}') # load model
    model.set_env(env, force_reset=True)

    # retrain the model (continuing from previous)
    for i in range(1, iters + 1):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=log_name)
        model.save(f"models/{name}-{int(steps) + i}")

# simulates how the model peforms after desired episode length
def simulate(model, env, tile, episodes=3, verbose=False):
    total = 0
    # create new sim for n episodes
    for ep in range(episodes):
        done = False
        obs = env.reset() # init

        # show info & env
        if verbose:
            print(f"Trial: {ep + 1}")
            env.render()

        # run sim until the agent is done (fails or wins)
        while not done:

            # predict based on env, then update env
            action, state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # tile number reached during episode
            if info["points"] >= tile:
                total += 1

            # show info & env
            if verbose:
                env.render() # show action made
                print(f'Oberservation: {obs}')
                print(f"Action: {action}\nReward: {reward}") # info
                print(f'Info: {info}')

    return total
    
        
if __name__ == "__main__":
    # Enviornment
    env = Env2048(size=4) # 4x4 2048 env

    # TRAINING
    # model = sb3.PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/") # Proximal Policy Optimization Algorithm
    # train(model, log_name="Agent", timesteps=10000, iters=5000) # 50 million runs in the game
    
    # RETRAINING
    # ppo = sb3.PPO.load("models/ppo_4x4-100x10^4")
    # retrain(env, "ppo-6824", timesteps=10000, iters=5000)

    # SIMULATING
    tag = 2940
    model = sb3.PPO.load(f"Agents/Agent-{tag}")
    total = simulate(model, env, points=1024, episodes=1000)
    print(total)