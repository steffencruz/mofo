from gym.envs.registration import register

register(
    id='MoFo-v0',
    entry_point='mofo.envs:MoFoEnv',
	max_episode_steps=200,
	reward_threshold=50.0
)
