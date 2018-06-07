
import tensorflow as tf
import numpy as np

def join_board(split_board_seq,nframes=1,input_dim=2):
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
    """Takes board[nsteps,nrows,ncols,1]
     and splits it into p1 and p2 binary boards
     returns [nsteps,nrows,ncols,2]"""

    if board_seq.get_shape()[0]>0:
        nframes=board_seq.get_shape()[0]
    else:
        print('Error: cannot split board without any frames')
        return 0

    split_board_seq = []
    for i in range(nframes): # loop over series of steps
        board = board_seq[i] if nframes>1 else board_seq
        split_b = np.zeros([board.shape[1],board.shape[2],2])

        # board p1 contains 0=empty,1=p1
        for j in range(2):
            if len(board[board==j+1]):
                ix,iy = np.where(board==j+1)
                split_b[ix,iy,j]=1
        print(split_b)
        split_board_seq.append(split_b)

    split_board_seq = np.array(split_board_seq)

    return split_board_seq

# experience replay
# advantage and action networks


class cnn_model():
    """Custom RL agent - The agent takes a state and produces an action."""

    def __init__(self,nrows,ncols,nactions,nfilters=8,f_size=4,h_size=10,d_rate=0.01,l_rate=0.01):
        """
            Current architecture is [using batch of size BS]:

            __Layer__                __Shape__             __Values__
            input_state         [BS,nrows,ncols,1]      0=empty,1=p1,2=p2

            __Filter__            __Filter Shape__
            conv_filter:     [f_size,f_size,1,nfilters]

            __Layer__                __Shape__
            conv_layer       [BS,nrows,ncols,nfilters]
            conv_flat        [BS,nrows*ncols*nfilters]
            hidden           [BS,10]
            output           [BS,ncols]

        """

        self.nactions = nactions
        self.eps = None
        self.eps_step = None
        self.eps_end = None

        # holders for summary objects that are relevant for different timescales
        self.step_summary = []
        self.game_summary = []
        self.batch_summary = []

        with tf.name_scope('Input'):
            self.input_state = tf.placeholder(shape=[None,nrows,ncols,1],dtype=tf.float32,name="input_state")
            self.game_summary.append(tf.summary.image('Input_State', self.input_state))
            tf.add_to_collection('input_layer',self.input_state)

        if nfilters>0:
            # convolutional layer [f_size*f_size from 1 input channel to nfilters output channels]
            with tf.name_scope('Convolution'):
                self.conv_filter = tf.Variable(tf.truncated_normal([f_size, f_size, 1, nfilters]),name="conv_filter")
                self.conv_layer = tf.nn.conv2d(self.input_state, self.conv_filter, strides=[1, 1, 1, 1], padding="SAME",name="conv_layer")

                with tf.name_scope('Activations'):
                    # self.conv_activations =  tf.transpose(self.conv_layer,perm=[3,0,1,2])
                    for i in range(nfilters):
                        act_layer = tf.reshape(self.conv_layer[-1,:,:,0],[-1,nrows,ncols,1])
                        self.game_summary.append(tf.summary.image('Filter'+str(i), act_layer))

                with tf.name_scope('Filters'):
                    self.conv_image = tf.transpose(self.conv_filter,perm=[3,0,1,2])
                    self.game_summary.append(tf.summary.image('Weights', self.conv_image))

            # for now let's ignore 2d output from conv and reshape the board into a 1D array
            self.conv_flat = tf.reshape(self.conv_layer, [-1, nrows * ncols * nfilters],name="conv_flat_layer")

            # hidden layer takes input state [nrows*ncols,1] and outputs [h_size,1]
            with tf.name_scope('Hidden'):
                self.hidden = tf.layers.dense(self.conv_flat,h_size,activation=tf.nn.relu,name="hidden_layer")
                self.game_summary.append(tf.summary.histogram('Hidden',self.hidden))
        else:
            # for now let's ignore 2d+3d info from input and reshape the board into a 1D array
            # also remove the empy third layer as this just complicates things
            self.input_flat = tf.reshape(self.input_state, [-1, nrows * ncols * 1],name="input_flat_layer")

            # hidden layer takes input state [nrows,ncols,3] and outputs [h_size,1]
            with tf.name_scope('Hidden'):
                self.hidden = tf.layers.dense(self.input_flat,h_size,activation=tf.nn.relu,name="hidden_layer")
                self.game_summary.append(tf.summary.histogram('Hidden',self.hidden))

        # dropout layer
        with tf.name_scope('Dropout'):
            self.dropout = tf.nn.dropout(self.hidden,1-d_rate)
            self.game_summary.append(tf.summary.histogram('Dropout',self.dropout))

        # output layer takes hidden layer [h_size,1] and outputs [a_size,1]
        with tf.name_scope('Output'):

            self.output = tf.layers.dense(self.hidden, nactions, activation=tf.nn.softmax,name="output_layer")
            tf.add_to_collection('action_probs',self.output)

            self.game_summary.append(tf.summary.histogram('Output',self.output))

            # chosen action is sampled from output probability distribution
            # self.sample_op = tf.multinomial(logits=self.output,num_samples=1,name="sample_op")
            # chosen action is action with largest probability
            self.sample_op = tf.argmax(self.output,1)
            tf.add_to_collection('sample_op',self.sample_op)

        # We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        with tf.name_scope('Rewards'):
            self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32,name='reward_holder')
            self.game_summary.append(tf.summary.histogram('Current_Reward_Batch',self.reward_holder))

        with tf.name_scope('Actions'):
            self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32,name='action_holder')
            self.game_summary.append(tf.summary.histogram('Current_Action_Batch',self.action_holder))

        with tf.name_scope('Loss'):
            self.onehot_labels=tf.one_hot(self.action_holder,nactions)
            self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels,
                                                                logits=self.output)
            # self.loss = tf.reduce_sum(self.reward_holder * self.cross_entropy)

            ##########################################################
            # old way of doing it [0] is batch size, [1] is action indx
            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

            tvars = tf.trainable_variables()

            self.gradient_holders = []
            for idx,var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss,tvars)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders,tvars))
            ##########################################################

            self.batch_summary.append(tf.summary.histogram('Cross_Entropy',self.cross_entropy))
            self.batch_summary.append(tf.summary.histogram('Loss',self.loss))

        with tf.name_scope('Train'):
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        with tf.name_scope('Performance'):
            self.running_reward = tf.placeholder(dtype=tf.float32,name='running_reward')
            self.step_summary.append(tf.summary.scalar('Running_Reward',self.running_reward))

            self.won   = tf.placeholder(dtype=tf.float32,name='record_won')
            self.lost  = tf.placeholder(dtype=tf.float32,name='record_lost')
            self.drew  = tf.placeholder(dtype=tf.float32,name='record_drew')
            self.steps = tf.placeholder(dtype=tf.float32,name='record_step')

            self.batch_summary.append(tf.summary.scalar('Won',self.won))
            self.batch_summary.append(tf.summary.scalar('Lost',self.lost))
            self.batch_summary.append(tf.summary.scalar('Drew',self.drew))
            self.batch_summary.append(tf.summary.scalar('Out-of-Steps',self.steps))

        print('\n\t-> Made agent\n\n')

    def get_summary(self,opt):

        if opt=='all':
            return tf.summary.merge_all()
        elif opt=='batch':
            return tf.summary.merge(self.batch_summary)
        elif opt=='game':
            return tf.summary.merge(self.game_summary)
        elif opt=='step':
            return tf.summary.merge(self.step_summary)
        else:
            print('Warning: a summary was not found for opt=',opt,'so full summary will be returned')
            return tf.summary.merge_all()

    def choose_action(self,sess,state,start_eps=0.0,end_eps=0.0,annealing_steps=10000):

        if self.eps==None or self.eps_step==None or self.eps_end==None:
            self.eps = start_eps
            self.eps_step = (start_eps-end_eps)/annealing_steps
            self.eps_end = end_eps

        if self.eps > np.random.rand():
            action = np.random.randint(0,self.nactions)
        else:
            action_probs = sess.run(self.output,{self.input_state:state})[0]
            vals = np.arange(self.nactions,dtype=np.int32)
            action = np.random.choice(vals,p=action_probs)

            # action = sess.run(self.sample_op,{self.input_state:state})[0][0]
            # if action>=self.nactions:
            # probs = sess.run(self.output,{self.input_state:state})
            # print('NN.output=',probs,'\t action chosen =',action)

        if self.eps>self.eps_end:
            self.eps -= self.eps_step

        return action

    # def get_train_stats(self):


class cnn_sb_model():
    """Custom RL agent - The agent takes a state,
    splits it into individual player boards and produces an action."""

    def __init__(self,nrows,ncols,nactions,nfilters=8,f_size=4,h_size=10,d_rate=0.01,l_rate=0.01):
        """
            Current architecture is [using batch of size BS]:

            __Layer__                __Shape__             __Values__
            input_state         [BS,nrows,ncols,1]      0=empty,1=p1,2=p2
            input_state_sep:    [BS,nrows,ncols,2]  0=empty,1=player[i+1] for layer i

            __Filter__            __Filter Shape__
            conv_filter:     [f_size,f_size,2,nfilters]

            __Layer__                __Shape__
            conv_layer       [BS,nrows,ncols,nfilters]
            conv_flat        [BS,nrows*ncols*nfilters]
            hidden           [BS,h_size]
            output           [BS,nactions]

        """

        self.nactions = nactions
        self.eps = None
        self.eps_step = None
        self.eps_end = None

        # holders for summary objects that are relevant for different timescales
        self.step_summary = []
        self.game_summary = []
        self.batch_summary = []

        with tf.name_scope('Input'):
            self.input_state = tf.placeholder(shape=[None,nrows,ncols,1],dtype=tf.float32,name="input_state")
            self.game_summary.append(tf.summary.image('Input_State', self.input_state))
            tf.add_to_collection('input_layer',self.input_state)

            # define split board first
            self.input_state_sep = tf.Variable(tf.zeros([1,nrows,ncols,2]),
                                    dtype=tf.float32,name='Input_State_Split')

            # now assign it to the output of split_board(states)
            tf.assign(self.input_state_sep,split_board(self.input_state),validate_shape=False)

        if nfilters>0:
            # convolutional layer [f_size*f_size from 1 input channel to nfilters output channels]
            with tf.name_scope('Convolution'):
                self.conv_filter = tf.Variable(tf.truncated_normal([f_size, f_size, 2, nfilters]),name="conv_filter")
                self.conv_layer = tf.nn.conv2d(self.input_state_sep, self.conv_filter, strides=[1, 1, 1, 1], padding="SAME",name="conv_layer")

                with tf.name_scope('Activations'):
                    # self.conv_activations =  tf.transpose(self.conv_layer,perm=[3,0,1,2])
                    for i in range(nfilters):
                        act_layer = tf.reshape(self.conv_layer[-1,:,:,0],[-1,nrows,ncols,1])
                        self.game_summary.append(tf.summary.image('Filter'+str(i), act_layer))

                with tf.name_scope('Filters'):
                    self.conv_image = tf.transpose(self.conv_filter,perm=[3,0,1,2])
                    self.game_summary.append(tf.summary.image('Weights', self.conv_image))

            # for now let's ignore 2d output from conv and reshape the board into a 1D array
            self.conv_flat = tf.reshape(self.conv_layer, [-1, nrows * ncols * nfilters],name="conv_flat_layer")

            # hidden layer takes input state [nrows*ncols,1] and outputs [h_size,1]
            with tf.name_scope('Hidden'):
                self.hidden = tf.layers.dense(self.conv_flat,h_size,activation=tf.nn.relu,name="hidden_layer")
                self.game_summary.append(tf.summary.histogram('Hidden',self.hidden))
        else:
            # for now let's ignore 2d+3d info from input and reshape the board into a 1D array
            # also remove the empy third layer as this just complicates things
            self.input_flat = tf.reshape(self.input_state, [-1, nrows * ncols * 1],name="input_flat_layer")

            # hidden layer takes input state [nrows,ncols,3] and outputs [h_size,1]
            with tf.name_scope('Hidden'):
                self.hidden = tf.layers.dense(self.input_flat,h_size,activation=tf.nn.relu,name="hidden_layer")
                self.game_summary.append(tf.summary.histogram('Hidden',self.hidden))

        # dropout layer
        with tf.name_scope('Dropout'):
            self.dropout = tf.nn.dropout(self.hidden,1-d_rate)
            self.game_summary.append(tf.summary.histogram('Dropout',self.dropout))

        # output layer takes hidden layer [h_size,1] and outputs [a_size,1]
        with tf.name_scope('Output'):

            self.output = tf.layers.dense(self.hidden, nactions, activation=tf.nn.softmax,name="output_layer")
            tf.add_to_collection('action_probs',self.output)

            self.game_summary.append(tf.summary.histogram('Output',self.output))

            # chosen action is sampled from output probability distribution
            # self.sample_op = tf.multinomial(logits=self.output,num_samples=1,name="sample_op")
            # chosen action is action with largest probability
            self.sample_op = tf.argmax(self.output,1)
            tf.add_to_collection('sample_op',self.sample_op)

        # We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        with tf.name_scope('Rewards'):
            self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32,name='reward_holder')
            self.game_summary.append(tf.summary.histogram('Current_Reward_Batch',self.reward_holder))

        with tf.name_scope('Actions'):
            self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32,name='action_holder')
            self.game_summary.append(tf.summary.histogram('Current_Action_Batch',self.action_holder))

        with tf.name_scope('Loss'):
            self.onehot_labels=tf.one_hot(self.action_holder,nactions)
            self.cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels,
                                                                logits=self.output)
            # self.loss = tf.reduce_sum(self.reward_holder * self.cross_entropy)

            ##########################################################
            # old way of doing it [0] is batch size, [1] is action indx
            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

            tvars = tf.trainable_variables()

            self.gradient_holders = []
            for idx,var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss,tvars)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders,tvars))
            ##########################################################

            self.batch_summary.append(tf.summary.histogram('Cross_Entropy',self.cross_entropy))
            self.batch_summary.append(tf.summary.histogram('Loss',self.loss))

        with tf.name_scope('Train'):
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        with tf.name_scope('Performance'):
            self.running_reward = tf.placeholder(dtype=tf.float32,name='running_reward')
            self.step_summary.append(tf.summary.scalar('Running_Reward',self.running_reward))

            self.won   = tf.placeholder(dtype=tf.float32,name='record_won')
            self.lost  = tf.placeholder(dtype=tf.float32,name='record_lost')
            self.drew  = tf.placeholder(dtype=tf.float32,name='record_drew')
            self.steps = tf.placeholder(dtype=tf.float32,name='record_step')

            self.batch_summary.append(tf.summary.scalar('Won',self.won))
            self.batch_summary.append(tf.summary.scalar('Lost',self.lost))
            self.batch_summary.append(tf.summary.scalar('Drew',self.drew))
            self.batch_summary.append(tf.summary.scalar('Out-of-Steps',self.steps))

        print('\n\t-> Made agent\n\n')

    def get_summary(self,opt):

        if opt=='all':
            return tf.summary.merge_all()
        elif opt=='batch':
            return tf.summary.merge(self.batch_summary)
        elif opt=='game':
            return tf.summary.merge(self.game_summary)
        elif opt=='step':
            return tf.summary.merge(self.step_summary)
        else:
            print('Warning: a summary was not found for opt=',opt,'so full summary will be returned')
            return tf.summary.merge_all()

    def choose_action(self,sess,state,start_eps=0.0,end_eps=0.0,annealing_steps=10000):

        if self.eps==None or self.eps_step==None or self.eps_end==None:
            self.eps = start_eps
            self.eps_step = (start_eps-end_eps)/annealing_steps
            self.eps_end = end_eps

        if self.eps > np.random.rand():
            action = np.random.randint(0,self.nactions)
        else:
            action_probs = sess.run(self.output,{self.input_state:state})[0]
            vals = np.arange(self.nactions,dtype=np.int32)
            action = np.random.choice(vals,p=action_probs)

            # action = sess.run(self.sample_op,{self.input_state:state})[0][0]
            # if action>=self.nactions:
            # probs = sess.run(self.output,{self.input_state:state})
            # print('NN.output=',probs,'\t action chosen =',action)

        if self.eps>self.eps_end:
            self.eps -= self.eps_step

        return action

    # def get_train_stats(self):
