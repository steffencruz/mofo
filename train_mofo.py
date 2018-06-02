import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

import my_models
import my_stats

# project checklist
# -> save the NNs and view the files
# -> get the model to train
# -> use self play with stronger opponent to train further, and also to determine which AI is better

if __name__ == '__main__':

    game_name = 'MoFo-v0'
    game = gym.make(game_name)

    observation = game.reset()
    nrows, ncols = game.observation_space.shape
    nactions = game.action_space.n

    total_episodes = 10000  # total number of games to train agent on
    episode = 0             # starting episode
    neval = 1000            # How often to store performance data
    update_frequency = 20   # How often to perform a training step.
    max_steps = nrows*ncols # Max moves per game for p1

    nfilters = 16           # number of convolutional filters
    f_size = 4              # convolutional filter sizes
    h_size = 10             # hidden layer size
    gamma = 0.95            # dicount reward rate
    d_rate = 0.01           # dropout rate
    l_rate = 0.01           # learning rate

    tf.reset_default_graph() # Clear the Tensorflow graph.

    NN = my_models.cnn_model(nrows=nrows,ncols=ncols,nactions=nactions,
                    nfilters=nfilters,f_size=f_size,h_size=h_size,
                    d_rate=d_rate,l_rate=l_rate)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    stats_logger = my_stats.training_log(nrows=nrows,ncols=ncols,max_steps=max_steps,
                        batch_size=update_frequency,num_episodes=total_episodes)

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        file_writer = tf.summary.FileWriter('./my-output'+'/graph-'+str(episode), sess.graph)

        while episode <= total_episodes:

            current_state = game.reset()
            episode += 1
            running_reward = 0.0

            for j in range(max_steps):

                state = current_state.copy()

                # choose action from NN
                action = NN.choose_action(sess,my_models.split_board(state))
                # apply action and update observation,reward,done,info
                next_state,reward,done,info = game.step(action)

                if j==max_steps-1:  # out-of-steps
                    info = -1 # flag the game as out-of-steps
                    reward-=10 # huge penalty to discourage this behaviour
                    # reward-=1 # small penalty to keep gradients stable
                    done = True # flag game as ended

                current_state = next_state
                running_reward += reward

                stats_logger.add_turn(current_state=current_state.copy(),
                                        reward=reward,action=action)

                # # write the running reward within each game
                # summary = sess.run(NN.running_reward_summary,
                #                     {NN.running_reward:float(running_reward)})
                # file_writer.add_summary(summary,episode)
                # file_writer.flush()

                if done == True: # end-of-game
                    # store game result [1=p1 win, 2=p2 win, 0=draw, -1=out-of-steps]
                    stats_logger.add_game(game_outcome=info,ep=episode,dr_gamma=gamma,norm_r=True)

                    raw_obs,sep_obs,actions,rewards = stats_logger.get_training_data()
                    # create dictionary of info for game -> new state is not included!
                    feed_dict={NN.input_state     : raw_obs,
                               NN.input_state_sep : sep_obs,
                               NN.action_holder   : actions,
                               NN.reward_holder   : rewards}
                    myloss = sess.run(NN.loss,feed_dict)

                    # stats_logger.add_game_loss(myloss)

                    if episode % update_frequency == 0: # every n=update_frequency games update network
                        # train on states, actions and rewards from last n=update_frequency games

                        tot,won,lost,drew,steps = stats_logger.get_performance_record(fetch_batches=-1,
                                                            sum_batches=True,percentage=True)
                        full_feed_dict = dict(feed_dict)
                        stats_dict = {NN.won:won, NN.lost:lost, NN.drew:drew, NN.steps:steps,
                                        NN.running_reward:float(running_reward)}
                        full_feed_dict.update(stats_dict)

                        fetches = [merged,NN.train_op]#,NN.cross_entropy,NN.loss]
                        summary = sess.run(fetches,feed_dict=full_feed_dict)[0]
                        file_writer.add_summary(summary,episode)
                        file_writer.flush()
                        print('** NN UPDATE    Ep: %4i'%episode,' Loss: %.2e'%myloss,'**',end='\r')

                    # Update our running tally of scores.
                    if episode % neval == 0:

                        nbat = round(neval/update_frequency)
                        fbat = np.arange(nbat)-nbat
                        # return the sum of the last nbat performance stats
                        tot,won,lost,drew,steps = stats_logger.get_performance_record(fetch_batches=fbat,
                                                            sum_batches=True,percentage=True)

                        print('EPISODE: %4i'%episode,'\t[ p-%3i'%tot,'  w-%.0f%%'%won,'  l-%.0f%%'%lost,'  d-%.0f%%'%drew,'  s-%.0f%% ]'%steps,sep='')

                        # file writer is needed for TensorBoard
                        file_writer = tf.summary.FileWriter('./my-output'+'/graph-'+str(episode), sess.graph)

                        # grab everything in fetches [data size is batch size]
                        fetches = [merged,
                                NN.input_state_sep,NN.conv_layer,NN.hidden,
                                NN.reward_holder,NN.action_holder,
                                NN.cross_entropy,NN.loss]

                        full_feed_dict = dict(feed_dict)
                        stats_dict = {NN.won:won, NN.lost:lost, NN.drew:drew, NN.steps:steps,
                                        NN.running_reward:float(running_reward)}
                        full_feed_dict.update(stats_dict)

                        summary = sess.run(fetches,feed_dict=full_feed_dict)[0]
                        file_writer.add_summary(summary,episode)
                        file_writer.flush()

                        # this saves checkpoints for re-opening
                        saver.save(sess,'./my-checkpoints'+'/model',global_step=episode)

                        # load in most recent model checkpoint
                        # game.env.InitAI()

                    break

    stats_logger.plot_stats(game_name)
    game.close()
