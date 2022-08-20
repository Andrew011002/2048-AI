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
def simulate(model, env, episodes=3):

    # create new sim for n episodes
    for ep in range(episodes):
        done = False
        obs = env.reset() # init
        env.render()
        print(f"Trial: {ep + 1}")

        # run sim until the agent is done (fails or wins)
        while not done:
            print(f'Oberservation: {obs}')
            action, state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(f"Action: {action}\nReward: {reward}") # info
            env.render() # show action made
            print(f'Info: {info}')
        
if __name__ == "__main__":
    # Enviornment
    env = Env2048(size=4) # 4x4 2048 env

    # TRAINING
    model = sb3.PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/") # Proximal Policy Optimization Algorithm
    train(model, log_name="Agent-Dync", timesteps=10000, iters=5000) # 50 million runs in the game
    
    # RETRAINING
    # ppo = sb3.PPO.load("models/ppo_4x4-100x10^4")
    # retrain(env, "ppo-6824", timesteps=10000, iters=5000)

    # SIMULATING
    # tag = 4049
    # model = sb3.PPO.load(f"models/Agent-{tag}")
    # simulate(model, env, episodes=1)