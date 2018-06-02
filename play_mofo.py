import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

# project checklist
# -> save the NNs and view the files
# -> get the model to train
# -> use self play with stronger opponent to train further, and also to determine which AI is better

def discount_rewards(r,gamma=0.95,normalize_rewards=False):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r,dtype=np.float32)
    # print('\nr  =',r)
    running_add = 0
    # print('Calculating discount rewards. r.size=',r.size,'r=',r)

    for t in reversed(range(0, r.size)):
        # print('running_add=',running_add,'running_add * gamma + r[t]=',running_add * gamma + r[t])
        running_add = running_add * gamma + r[t]
        discounted_r[t] = float(running_add)
        # print('dr[ %2i'%t,'] = %.3f'%running_add,' \t\trunning_add { =%.3f'%running_add,'} * gamma { =%.2f'%gamma,'} + r[ %2i'%t,'] { =%.2f'%r[t],'}')

    # print('\n->discounted_r',discounted_r)
    if normalize_rewards and not np.all(discounted_r==0):
        # print('could not normalize rewards because sum=%.3f'%dr_mean,'+/-%.3f'%dr_stddev,'\ndiscounted_r=',*discounted_r)
        # print('discounted rewards before norm: avg=%.3f'%dr_mean,'+/-%.3f'%dr_stddev,end=' ')
        dd = discounted_r
        dd -= np.mean(discounted_r)
        dd /= np.std(discounted_r)
        discounted_r = dd
        # print('\t after norm: avg=%.3f'%np.sum(discounted_r),'+/-%.3f'%np.std(discounted_r))

    return discounted_r

def join_board(split_board_seq,nframes=1,input_dim=3):
    """Takes board[nrows,ncols,nsteps]
     and splits it into p1 and p2 binary boards
     returns splitBoard[nrows,ncols,nsteps*2]"""

    board_seq = []
    for i in range(nframes): # loop over series of steps

        # pull out i'th frame if multiple frames
        split_b = split_board_seq[i,:,:,:] if nframes>1 else split_board_seq

        board = np.zeros([split_b.shape[0],split_b.shape[1],1])

        for j in range(input_dim-1): board[split_b[:,:,j]==1] = j+1

        board_seq.append(board)

    board_seq = np.array(board_seq)

    return board_seq

def split_board(board_seq,nframes=1):
    """Takes board[nrows,ncols,nsteps]
     and splits it into p1 and p2 binary boards
     returns splitBoard[nrows,ncols,nsteps*2]"""

    # print('\n\nboard_seq=\n',board_seq,'\nboard_seq.shape=',board_seq.shape)
    split_board_seq = []
    for i in range(nframes): # loop over series of steps
        board = board_seq[i] if nframes>1 else board_seq
        # print('i=',i,'\nboard=\n',board)
        split_b = np.zeros([board.shape[0],board.shape[1],3])
        # board p1 contains 0=empty,1=p1
        for j in range(3):
            if len(board[board==j+1]):
                ix,iy = np.where(board==j+1)
                split_b[ix,iy,j]=1

        split_board_seq.append(split_b)

    split_board_seq = np.array(split_board_seq)

    return split_board_seq

class cnn_model():
    def __init__(self,nrows,ncols,nactions,nfilters=8,f_size=4,h_size=10,d_rate=0.01,l_rate=0.01):
        """Custom RL agent - The agent takes a state and produces an action.
            Current architecture is [using batch of size BS]:

            __Layer__                __Shape__           __Values__
            input_state         [BS,nrows,ncols,1]  0=empty,1=p1,2=p2
            input_state_sep:    [BS,nrows,ncols,3]  0=empty,1=player[i+1] for layer i

            __Filter__            __Filter Shape__
            conv_filter:     [f_size,f_size,3,nfilters]

            __Layer__                __Shape__
            conv_flat           [BS,nrows,ncols,1]

        """

        self.nactions = nactions
        self.eps = None
        self.eps_step = None

        with tf.name_scope('Input'):
            with tf.name_scope('Raw_Input'):
                self.input_state = tf.placeholder(shape=[None,nrows,ncols,1],dtype=tf.float32,name="input_state")
                tf.summary.image('Input_State', self.input_state)

            with tf.name_scope('Split_Input'):
                self.input_state_sep = tf.placeholder(shape=[None,nrows,ncols,3],dtype=tf.float32,name="input_state_sep")
                # would be nice to use split_board internally as a class method
                # self.input_state_sep = self.split_board(self.input_state)
                tf.summary.image('Input_State_Sep', self.input_state_sep)

        # convolutional layer [f_size*f_size from 3 input channels to nfilters output channels]
        with tf.name_scope('Convolution'):
            self.conv_filter = tf.Variable(tf.truncated_normal([f_size, f_size, 3, nfilters]),name="conv_filter")
            self.conv_layer = tf.nn.conv2d(self.input_state_sep, self.conv_filter, strides=[1, 1, 1, 1], padding="SAME",name="conv_layer")
            with tf.name_scope('Filters'):
                self.conv_image = tf.transpose(self.conv_filter,perm=[3,0,1,2])
                # create separate weight slices for the 3 split input states
                self.conv_image1 = tf.reshape(self.conv_image[:,:,:,0],[-1,f_size,f_size,1]) # weights that respond to layer 0 [p1]
                self.conv_image2 = tf.reshape(self.conv_image[:,:,:,1],[-1,f_size,f_size,1]) # weights that respond to layer 1 [p2]
                self.conv_image0 = tf.reshape(self.conv_image[:,:,:,2],[-1,f_size,f_size,1]) # weights that respond to layer 2 [empty]

                tf.summary.image('Weights_RGB', self.conv_image)
                tf.summary.image('Weights_p1', self.conv_image1)
                tf.summary.image('Weights_p2', self.conv_image2)
                tf.summary.image('Weights_p0', self.conv_image0)

            with tf.name_scope('Activations'):
                self.conv_activations =  tf.transpose(self.conv_layer,perm=[3,0,1,2])
                tf.summary.image('Activations', self.conv_activations)

        # for now let's ignore 2d output from conv and reshape the board into a 1D array
        self.conv_flat = tf.reshape(self.conv_layer, [-1, nrows * ncols * nfilters],name="conv_flat_layer")

        # hidden layer takes input state [nrows*ncols,1] and outputs [h_size,1]
        with tf.name_scope('Hidden'):
            self.hidden = tf.layers.dense(self.conv_flat,h_size,activation=tf.nn.relu,name="hidden_layer")
            tf.summary.histogram('Hidden',self.hidden)

        # dropout layer
        with tf.name_scope('Dropout'):
            self.dropout = tf.nn.dropout(self.hidden,1-d_rate)
            tf.summary.histogram('Dropout',self.dropout)

        # output layer takes hidden layer [h_size,1] and outputs [a_size,1]
        with tf.name_scope('Output'):

            self.output = tf.layers.dense(self.hidden, nactions, activation=tf.nn.softmax,name="output_layer")
            tf.add_to_collection('action_probs',self.output)

            tf.summary.histogram('Output',self.output)

            # chosen action is sampled from output probability distribution
            self.sample_op = tf.multinomial(logits=self.output,num_samples=1,name="sample_op")
            tf.add_to_collection('sample_op',self.sample_op)
            # chosen action is action with largest probability
            # chosen_action = tf.argmax(output,1)

        # We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        with tf.name_scope('Rewards'):
            self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32,name='reward_holder')
            tf.summary.histogram('Current_Reward_Batch',self.reward_holder)

        with tf.name_scope('Actions'):
            self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32,name='action_holder')
            tf.summary.histogram('Current_Action_Batch',self.action_holder)

        with tf.name_scope('Loss'):
            self.onehot_labels=tf.one_hot(self.action_holder,nactions)
            self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels,
                                                                logits=self.output)
            self.loss = tf.reduce_sum(self.reward_holder * self.cross_entropy)
            tf.summary.histogram('Cross_Entropy',self.cross_entropy)
            tf.summary.histogram('Loss',self.loss)

        with tf.name_scope('Train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        with tf.name_scope('Performance'):

            self.running_reward = tf.placeholder(dtype=tf.float32,name='running_reward')
            self.running_reward_summary = tf.summary.scalar('Running_Reward',self.running_reward)

            self.won   = tf.placeholder(dtype=tf.float32,name='record_won')
            self.lost  = tf.placeholder(dtype=tf.float32,name='record_lost')
            self.drew  = tf.placeholder(dtype=tf.float32,name='record_drew')
            self.steps = tf.placeholder(dtype=tf.float32,name='record_step')
            tf.summary.scalar('Won',self.won)
            tf.summary.scalar('Lost',self.lost)
            tf.summary.scalar('Drew',self.drew)
            tf.summary.scalar('Out-of-Steps',self.steps)

        print('\n\t-> Made agent\n\n')

    def choose_action(self,split_state,annealing_steps=10000,start_eps=0.5,end_eps=0.01):

        if self.eps==None or self.eps_step==None:
            self.eps = start_eps
            self.eps_step = (start_eps-end_eps)/annealing_steps

        if self.eps > np.random.rand():
            action = np.random.randint(0,self.nactions)
        else:
            action = sess.run(self.sample_op,
                {self.input_state_sep:split_state})[0][0]

        self.eps -= self.eps_step

        return action

    # def get_train_stats(self):

class training_log():
    """
        Takes care of organizing data for each game and batches of game

        Instructions:
            1. Call add_turns after each step to store turn info
            2. At the end of each game call add_games to store game info
            3. When batch_size games have been logged, add_batch will be called
                automatically and will create a summary for whole batch
            4. To produce a dictionary for training, call get_training_data
            5. Call get_performance_record to get played-won-lost-drew-steps stats
            6.
    """
    def __init__(self,nrows,ncols,max_steps,batch_size,num_episodes):
        self.nrows = nrows
        self.ncols = ncols
        self.max_turns = max_steps
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.max_batches = int(num_episodes/batch_size)

        self.reset_batch_log()

    def add_turn(self,current_state,action,reward,next_state=None):
        """Adds information about current turn to turns log"""

        if self.nturns==self.max_turns:
            print('Warning: game length is',self.nturns+1,'which exceeds max_steps [',self.max_turns,'] call add_game to make new game.')
            return 0
        if self.ngames==self.batch_size:
            # print('Info: batch',self.nbatches+1,'/',self.max_batches,'is complete. Storing info and making new batch.')
            self.reset_game_log()

        self.turns_cstate[self.nturns,:,:] = np.array(current_state)
        self.turns_action[self.nturns] = action
        self.turns_reward[self.nturns] = reward

        if next_state!=None:
            self.turns_nstate[self.nturns,:,:] = np.array(next_state)

        # print('Batch:',self.nbatches+1,'\tGame:',self.ngames+1,'\tTurn:',self.nturns+1)
        self.nturns+=1

        return self.nturns

    def add_game(self,game_outcome,ep,dr_gamma=0.95,norm_r=True):
        """Adds information about current game to games log"""

        # if self.total_in_batch==0:
        #     self.reset_game_log()
        if self.nturns==0:
            print('Error: game had zero moves. Nothing was added to game_log')
            return 0
        if self.ngames==self.batch_size:
            print('Error: trying to add game number',self.ngames+1,'to batch',self.nbatches+1,'[ batch size =',self.batch_size,']')
            return 0

        i0 = self.total_in_batch
        discounted_rewards = discount_rewards(self.turns_reward,gamma,norm_r)
        for i in range(self.nturns):
            self.games_cstate[i0+i,:,:] = self.turns_cstate[i,:,:]
            self.games_action[i0+i] = self.turns_action[i]
            self.games_reward[i0+i] = discounted_rewards[i]

        self.games_total_reward[self.ngames] = np.sum(self.turns_reward)
        self.games_length[self.ngames] = self.nturns
        self.games_record[self.ngames] = game_outcome

        # update total batch size by length of last game
        self.total_in_batch+=self.nturns
        self.ngames+=1

        # clear turn logs
        self.reset_turn_log()

        if self.ngames==self.batch_size:
            # print('Info: batch',self.nbatches+1,'/',self.max_batches,'is complete. Storing info and making new batch.')
            self.add_batch()

        return self.ngames

    def add_batch(self):
        """Adds summary information across multiple batches to batch log
            - Note that calling this function ends the batch and deletes game info
        """

        if self.total_in_batch==0:
            print('Error: batch has zero games. Nothing was added to batch_log')
            return 0
        if self.ngames!=self.batch_size:
            print('Warning: batch has size',self.ngames+1,'but size',self.batch_size,'was expected..')


        unique, counts = np.unique(self.games_record, return_counts=True)
        self.batch_record[self.nbatches,0] = len(self.games_record)
        key = [1,2,0,-1]
        for i,k in enumerate(key):
            if len(counts[unique==k]):
                self.batch_record[self.nbatches,i+1] = int(counts[unique==k])

        iend = self.total_in_batch
        self.batch_ave_reward[self.nbatches] = np.mean(self.games_total_reward[0:iend])
        self.batch_std_reward[self.nbatches] = np.std(self.games_total_reward[0:iend])

        self.batch_ave_turns[self.nbatches] = np.mean(self.games_length[0:iend])
        self.batch_std_turns[self.nbatches] = np.std(self.games_length[0:iend])

        # self.batch_ave_loss[self.nbatches] = np.mean(self.games_loss[0:iend])
        # self.batch_std_loss[self.nbatches] = np.std(self.games_loss[0:iend])

        self.nbatches+=1

    def get_training_data(self):

        raw_states = np.zeros([self.total_in_batch,self.nrows,self.ncols,1])
        for i in range(self.total_in_batch):
            raw_states[i,:,:,0] = self.games_cstate[i,:,:]

        # appropriate dimensions are taken care of in split_board with argument self.batch
        sep_states = split_board(self.games_cstate,self.total_in_batch)

        actions = self.games_action[0:self.total_in_batch]
        rewards = self.games_reward[0:self.total_in_batch]

        return raw_states,sep_states,actions,rewards

    def get_performance_record(self,fetch_batches=[-1],percentage=True,sum_batches=False):
        """ returns game performance stats
        eg. won 70, lost 3, drew 25, out-of-steps 2, played 100
        stored as elements in stats:- 0=w,1=l,2=d,3=s,4=tot
        returns: summary stats for all batches in fetch_batches
        """
        # if these results are already stored it is easy to sum multiple batches

        if hasattr(fetch_batches,"__len__"):
            nfetch = len(fetch_batches)
        else:
            nfetch=1
            fetch_batches = [fetch_batches]

        stats = np.zeros([nfetch,5])
        for i,bat in enumerate(fetch_batches):
            indx = 0
            if np.abs(bat)>self.nbatches+1:
                print('Error: cannot get performance stats for batch ',bat,'.. [ max =',self.nbatches,']')
                return stats
            elif bat<0:
                indx = self.nbatches+bat
            else:
                indx = bat

            stats[i,:] = self.batch_record[indx,:]

        if sum_batches:
            stats = np.sum(stats,axis=0)

        tot,won,lost,drew,step = [],[],[],[],[]
        if percentage:
            if sum_batches:
                stats[1:]*=100.0/stats[0]
                tot,won,lost,drew,step = stats
            else:
                for j in range(nfetch):
                    stats[j,1:]*=100.0/stats[j,0]
                tot  = stats[:,0]
                won  = stats[:,1]
                lost = stats[:,2]
                drew = stats[:,3]
                step = stats[:,4]

        return tot,won,lost,drew,step

    def reset_turn_log(self):
        self.nturns=0

        # pre-declared numpy containers that can contain up to max_steps elements
        self.turns_cstate = np.zeros([self.max_turns,self.nrows,self.ncols])
        self.turns_nstate = np.zeros([self.max_turns,self.nrows,self.ncols])
        self.turns_action = np.zeros(self.max_turns)
        self.turns_reward = np.zeros(self.max_turns)

    def reset_game_log(self):
        self.ngames=0
        self.total_in_batch=0

        max_batch = self.max_turns*self.batch_size
        # pre-declared numpy containers that can contain up to max_batch elements
        self.games_cstate = np.zeros([max_batch,self.nrows,self.ncols])
        self.games_nstate = np.zeros([max_batch,self.nrows,self.ncols])
        self.games_action = np.zeros(max_batch)
        self.games_reward = np.zeros(max_batch)

        self.games_total_reward = np.zeros(self.batch_size)
        self.games_record = np.zeros(self.batch_size)
        self.games_length = np.zeros(self.batch_size)

        self.reset_turn_log()

    def reset_batch_log(self):
        self.nbatches=0
        self.batch_record = np.zeros([self.max_batches,5])
        self.batch_ave_reward = np.zeros(self.max_batches)
        self.batch_std_reward = np.zeros(self.max_batches)
        self.batch_ave_turns = np.zeros(self.max_batches)
        self.batch_std_turns = np.zeros(self.max_batches)

        self.reset_game_log()

    def plot_stats(self,game_name):

        # plot variables which coincide with episodes [eps] array
        eps = range(0,self.num_episodes,self.batch_size)

        fig = plt.figure(figsize=(10,8), dpi=90)
        fig.patch.set_facecolor('white')
        fig.suptitle(game_name+' Training', fontsize=20, fontweight='bold')

        ax = fig.add_subplot(221)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Average Score [error band = stderr]', fontsize=14)
        plt.plot(eps,self.batch_ave_reward,'k-')
        plt.fill_between(eps, self.batch_ave_reward-self.batch_std_reward,
                    self.batch_ave_reward+self.batch_std_reward,alpha=0.2)

        ax = fig.add_subplot(222)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Average Loss', fontsize=14)
        # plt.plot(eps,self.batch_ave_,'k-')
        # plt.fill_between(eps, avg_l-std_l, avg_l+std_l,color='r',alpha=0.2)

        ax = fig.add_subplot(223)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Average Turns', fontsize=14)
        plt.plot(eps,self.batch_ave_turns,'k-')
        plt.fill_between(eps, self.batch_ave_turns-self.batch_std_turns,
                    self.batch_ave_turns+self.batch_std_turns,color='g',alpha=0.2)

        ax = fig.add_subplot(224)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Performance per batch', fontsize=14)

        _,won,lost,drew,step = self.get_performance_record(fetch_batches=np.arange(self.nbatches),
                                                            sum_batches=False,percentage=True)

        plt.plot(eps,won,'g-',label='won')
        plt.plot(eps,lost,'r-',label='lost')
        plt.plot(eps,drew,'b-',label='drew')
        plt.plot(eps,step,'k:',label='out-of-steps')
        plt.legend(fontsize=9)

        # plt.show()
        plt.pause(30)
        plt.savefig('training_'+game_name)

if __name__ == '__main__':

    game_name = 'MoFo-v0'
    game = gym.make(game_name)

    # custom initialization
    # nrows, ncols = 7,7
    # game.env.initialize(nrows,ncols)

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

    NN = cnn_model(nrows=nrows,ncols=ncols,nactions=nactions,
                    nfilters=nfilters,f_size=f_size,h_size=h_size,
                    d_rate=d_rate,l_rate=l_rate)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    stats_logger = training_log(nrows=nrows,ncols=ncols,max_steps=max_steps,
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
                action = NN.choose_action(split_board(state))
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

                    #Update our running tally of scores.
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
