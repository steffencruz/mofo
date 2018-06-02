
import tensorflow as tf
import numpy as np


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

class cnn_model():
    """Custom RL agent - The agent takes a state and produces an action."""

    def __init__(self,nrows,ncols,nactions,nfilters=8,f_size=4,h_size=10,d_rate=0.01,l_rate=0.01):
        """
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

        # holders for summary objects that are relevant for different timescales
        self.step_summary = []
        self.game_summary = []
        self.batch_summary = []

        with tf.name_scope('Input'):
            with tf.name_scope('Raw_Input'):
                self.input_state = tf.placeholder(shape=[None,nrows,ncols,1],dtype=tf.float32,name="input_state")
                self.game_summary.append(tf.summary.image('Input_State', self.input_state))

            with tf.name_scope('Split_Input'):
                self.input_state_sep = tf.placeholder(shape=[None,nrows,ncols,3],dtype=tf.float32,name="input_state_sep")
                tf.add_to_collection('input_layer',self.input_state_sep)

                # would be nice to use split_board internally as a class method
                # self.input_state_sep = self.split_board(self.input_state)
                self.game_summary.append(tf.summary.image('Input_State_Sep', self.input_state_sep))

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

                self.game_summary.append(tf.summary.image('Weights_RGB', self.conv_image))
                self.game_summary.append(tf.summary.image('Weights_p1', self.conv_image1))
                self.game_summary.append(tf.summary.image('Weights_p2', self.conv_image2))
                self.game_summary.append(tf.summary.image('Weights_p0', self.conv_image0))

            with tf.name_scope('Activations'):
                self.conv_activations =  tf.transpose(self.conv_layer,perm=[3,0,1,2])
                self.game_summary.append(tf.summary.image('Activations', self.conv_activations))

        # for now let's ignore 2d output from conv and reshape the board into a 1D array
        self.conv_flat = tf.reshape(self.conv_layer, [-1, nrows * ncols * nfilters],name="conv_flat_layer")

        # hidden layer takes input state [nrows*ncols,1] and outputs [h_size,1]
        with tf.name_scope('Hidden'):
            self.hidden = tf.layers.dense(self.conv_flat,h_size,activation=tf.nn.relu,name="hidden_layer")
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
            self.sample_op = tf.multinomial(logits=self.output,num_samples=1,name="sample_op")
            tf.add_to_collection('sample_op',self.sample_op)
            # chosen action is action with largest probability
            # chosen_action = tf.argmax(output,1)

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
            self.loss = tf.reduce_sum(self.reward_holder * self.cross_entropy)

            self.batch_summary.append(tf.summary.histogram('Cross_Entropy',self.cross_entropy))
            self.batch_summary.append(tf.summary.histogram('Loss',self.loss))

        with tf.name_scope('Train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
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

    def choose_action(self,sess,split_state,annealing_steps=10000,start_eps=0.5,end_eps=0.01):

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
