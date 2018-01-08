import gym
import gym_reacher2
env = gym.make("Reacher2-v0")
env.env._init(
    arm0 = .05,    # length of limb 1
    arm1 = .2,     # length of limb 2
    torque0 = 100, # torque of joint 1
    torque1 = 400,  # torque of joint 2
    fov=70,        # field of view
    colors={
            "arenaBackground": ".9 .0 .5",
            "arenaBorders": "0.1 0.1 0.4",
            "arm0": "0.8 0.7 0.1",
            "arm1": "0.2 0.5 0.1"
        },
    topDown=True   # top-down centered camera?
)
env.reset()
for i in range(10000):
    env.render()
    env.step(env.action_space.sample())

env.close()