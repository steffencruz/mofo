# MoFo [More Four]
A twist on the classic connect-4 game, where connecting 4 counters in a straight line scores a point and earns the player another turn. With a strong playing strategy, a player can dominate the board their opponent by accumulating rewards and additional turns. The winner is the player with the most points when the board is full. This can be played with any board size or shape, which significantly affects the complexity of the game and possible strategies. 

The training code stores the variables and graphs so that they can be analyzed in TensorBoard. They can also be loaded back into the game as an AI opponent using the PlayGame method of mofo.py

This game, imported as a custom OpenAI Gym environment, provides a rich testing ground for training an RL agent to play the game to 
study the dependence of AI performance on differen model architectures and board shapes.
