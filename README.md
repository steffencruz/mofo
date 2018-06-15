# MoFo [More Four]
A twist on the classic connect-4 game, where connecting 4 counters in a straight line scores a point and earns the player another turn. With a strong playing strategy, a player can dominate the board their opponent by accumulating rewards and additional turns. The winner is the player with the most points when the board is full. This can be played with any board size or shape, which significantly affects the complexity of the game and possible strategies.

The training code stores the variables and graphs so that they can be analyzed in TensorBoard. They can also be loaded back into the game as an AI opponent using the PlayGame method of mofo.py

This game, imported as a custom OpenAI Gym environment, provides a rich testing ground for training an RL agent to play the game to 
study the dependence of AI performance on different model architectures and board shapes.

To play in OpenAI Gym:-

    1. Install OpenAI gym at https://github.com/openai/gym

    2. Move this mofo.py file to gym/envs/my_collection

    3. create or add to existing __init__.py file in gym/envs/my_collection:

        from gym.envs.my_collection.mofo import MoFoEnv

    4. register the env in __init__.py file in gym/envs/

        register(
            id='MoFo-v0',
            entry_point='gym.envs.my_collection:MoFoEnv',
            max_episode_steps=200,
            reward_threshold=50.0,
        )

    5. To load environment and apply custom settings;

        game = gym.make('MoFo-v0') # default initialization
        game.env.initialize(nrows,ncols,verbose,training,testing)
        # all variables and methods are accessible via game.env
