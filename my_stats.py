import matplotlib.pyplot as plt
import numpy as np
import my_models

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
        discounted_rewards = my_models.discount_rewards(self.turns_reward,dr_gamma,norm_r)
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
        sep_states = my_models.split_board(self.games_cstate,self.total_in_batch)

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
