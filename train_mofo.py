"""
    More-Four game trainer, made by Steffen Cruz 2018

    Trains an RL agent to play mofo.

    The opponent is np.random.randint() but can be set to an earlier version
    of the model by setting update_opponent=True

    -> Results are written every $neval games to my-output/graph-$neval
    -> To view trained model in tensorboard, tensorboard --logdir my-output/graph-$neval/

    TO DO:
    -> Make it train better.
    -> Make a tournament
"""

import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

import my_models
import my_stats

def format_array(arr,sep=' ',eol='',dtype='float'):
    strarr=''
    for i in range(len(arr)):
        if dtype=='int':
            strarr+='{:4.0f}'.format(arr[i])+sep
        elif dtype=='float':
            strarr+='{:.2f}'.format(arr[i])+sep
    strarr+=eol
    return strarr

if __name__ == '__main__':

    game_name = 'MoFo-v0'
    game = gym.make(game_name)

    observation = game.reset()
    nrows, ncols = game.observation_space.shape
    nactions = game.action_space.n

    total_episodes = 10000  # total number of games to train agent on
    episode = 0             # starting episode
    max_steps = nrows*ncols # Max moves per game for p1
    batch_size = 10         # How often to perform a training step.
    neval = 500             # How often to store performance data

    update_opponent=False   # if true the NN will train against the most recent version of itself
    save_model=False        # if true the model is saved every neval games
    debug=False             # prints a bunch of stuff (~everything) to help debugging

    nfilters = 8            # number of convolutional filters
    f_size = 2              # convolutional filter sizes
    h_size = 20             # hidden layer size
    gamma = 0.0             # dicount reward rate -> depends on game length
    d_rate = 0.00           # dropout rate
    l_rate = 0.01           # learning rate

    tf.reset_default_graph() # Clear the Tensorflow graph.

    NN = my_models.cnn_model(nrows=nrows,ncols=ncols,nactions=nactions,
                    nfilters=nfilters,f_size=f_size,h_size=h_size,
                    d_rate=d_rate,l_rate=l_rate)

    merged        = NN.get_summary('all')
    step_summary  = NN.get_summary('step')
    game_summary  = NN.get_summary('game')
    batch_summary = NN.get_summary('batch')

    init = tf.global_variables_initializer()
    # saver is needed to save and restore model
    saver = tf.train.Saver()

    stats_logger = my_stats.training_log(nrows=nrows,ncols=ncols,max_steps=max_steps,
                        batch_size=batch_size,num_episodes=total_episodes)

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        # file writer is needed for TensorBoard
        file_writer = tf.summary.FileWriter('./my-output'+'/graph-'+str(episode), sess.graph)
        start_time = time.time()

        # gradients are set to 0 for trainable variables
        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while episode <= total_episodes:

            current_state = game.reset()
            episode += 1
            running_reward = 0.0

            for j in range(max_steps):

                current_state_copy = current_state.copy()
                state = np.reshape(current_state,(1,nrows,ncols,1)).copy()

                # choose action from NN
                # action = NN.choose_action(sess,state)
                action_probs = sess.run(NN.output,{NN.input_state:state})[0]
                action = np.random.choice(np.arange(ncols,dtype=np.int32),p=action_probs)

                # apply action and update observation,reward,done,info
                next_state,reward,done,info = game.step(action)

                if debug:
                    print('\nEpisode:',episode,'turn:',j,'action:',action,'action_probs: ',format_array(action_probs),'reward:',reward)

                if j==max_steps-1:  # out-of-steps
                    info = -1 # flag the game as out-of-steps
                    # reward-=10 # huge penalty to discourage this behaviour
                    reward-=1 # small penalty to keep gradients stable
                    done = True # flag game as ended

                stats_logger.add_turn(current_state=current_state_copy,
                                        reward=reward,action=action)

                current_state = next_state
                running_reward += reward

                if done == True: # end-of-game
                    # store game result [1=p1 win, 2=p2 win, 0=draw, -1=out-of-steps]
                    stats_logger.add_game(game_outcome=info,ep=episode,dr_gamma=gamma,norm_r=False)

                    # grab training data for most recent fame
                    observations,actions,rewards = stats_logger.get_training_data(ngames=1)
                    # create dictionary of info for game -> new state is not included!
                    feed_dict={NN.input_state     : observations,
                               NN.action_holder   : actions,
                               NN.reward_holder   : rewards}

                    if debug:
                        ins,acts,rews,indexes,resp,loss,outputs =sess.run([NN.input_state,NN.action_holder,NN.reward_holder,NN.indexes,NN.responsible_outputs,NN.loss,NN.output],feed_dict=feed_dict)
                        print('\naction_holder    =',format_array(acts,dtype='int'))
                        print('reward_holder    =',format_array(rews,dtype='int'))
                        print('indexes          =',format_array(indexes,dtype='int'))
                        print('resp_outputs     =',format_array(resp))
                        print('log(resp)        =',format_array(np.log(resp)))
                        print('rewards*log(resp)=',format_array(np.log(resp)*rewards))
                        print('---> mean = %.2f'%np.mean(np.log(resp)*rewards),'\nloss=',loss)
                        print('inputs =',ins.reshape((-1,1,nrows,ncols)))
                        print('outputs =',outputs)

                    game_loss,game_ce = sess.run([NN.loss,NN.cross_entropy],feed_dict)
                    stats_logger.add_game_performance((episode%batch_size),game_loss,game_ce)

                    # gradients for each layer are calculated each game
                    grads = sess.run(NN.gradients, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad

                    if episode % batch_size == 0: # every n=batch_size games update network

                        my_dict = dict(zip(NN.gradient_holders, gradBuffer))
                        _ = sess.run(NN.update_batch,feed_dict=my_dict)

                        # after performing a network update (training step) we reset the gradients
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                        if save_model:
                            # write the running reward for this game to a scalar for TensorBoard
                            rr = stats_logger.get_running_reward()
                            for i,R in enumerate(rr):
                                summary = sess.run(step_summary,{NN.running_reward:R})
                                file_writer.add_summary(summary,episode-batch_size+i)

                            # get stats for this batch
                            tot,won,lost,drew,steps = stats_logger.get_batch_record(fetch_batches=-1,
                                                                sum_batches=False,percentage=True)

                            fetches = [game_summary,NN.train_op]
                            full_feed_dict = dict(feed_dict)
                            stats_dict = {NN.won:won, NN.lost:lost, NN.drew:drew, NN.steps:steps}
                            full_feed_dict.update(stats_dict)

                            summary = sess.run(fetches,feed_dict=full_feed_dict)[0]
                            file_writer.add_summary(summary,episode)
                            file_writer.flush()

                        print('** NN UPDATE    Ep: %4i'%episode,' Loss: %.2e'%game_loss,
                                ' Cross Entropy: %.2e'%game_ce,'**',end='\r')

                    # Update our running tally of scores and create a new output file.
                    if episode % neval == 0:

                        nbat = round(neval/batch_size)
                        fbat = np.arange(nbat)-nbat
                        # return the overall performance stats of the last nbat games
                        tot,won,lost,drew,steps = stats_logger.get_batch_record(fetch_batches=fbat,
                                                            sum_batches=True,percentage=True)
                        # return the avg and std reward of the last nbat games
                        avg_rew,std_rew = stats_logger.get_batch_rewards(fetch_batches=fbat,
                                                            sum_batches=True)

                        end_time = time.time()
                        print('EPISODE: %4i'%episode,'\t[ p-%3i'%tot,'  w-%3.0f%%'%won,
                            '  l-%3.0f%%'%lost,'  d-%3.0f%%'%drew,'  s-%3.0f%% ]'%steps,
                            '  Avg. Reward = %.1f'%avg_rew,' +/- %.1f'%std_rew,
                            '  Time = %.2f s'%(end_time-start_time),sep='')
                        start_time = time.time()

                        if save_model:
                            file_writer = tf.summary.FileWriter('./my-output'+'/graph-'+str(episode), sess.graph)

                            # grab everything in fetches [data size is batch size]
                            fetches = [batch_summary,NN.input_state,NN.hidden,
                                                     NN.reward_holder,NN.action_holder,
                                                     NN.cross_entropy,NN.loss]
                            if nfilters>0: fetches.append(NN.conv_layer)

                            full_feed_dict = dict(feed_dict)
                            stats_dict = {NN.won:won, NN.lost:lost, NN.drew:drew, NN.steps:steps,
                                            NN.running_reward:float(running_reward)}
                            full_feed_dict.update(stats_dict)

                            summary = sess.run(fetches,feed_dict=full_feed_dict)[0]
                            file_writer.add_summary(summary,episode)
                            file_writer.flush()

                            # this saves checkpoints for re-opening
                            saver.save(sess,'./my-checkpoints'+'/model',global_step=episode)

                            if update_opponent: # load most recent model checkpoint as opponent
                                game.env.InitAI('./my-checkpoints')

                    break

    npts = 100
    ngroup = float(total_episodes)/float(batch_size*npts)
    stats_logger.plot_stats(game_name,ngroup=ngroup)
    game.close()
