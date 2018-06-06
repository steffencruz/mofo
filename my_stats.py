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

    def add_game(self,game_outcome,ep,dr_gamma=0.95,norm_r=False):
        """Adds information about current game to games log"""

        if self.nturns==0:
            print('Error: game had zero moves. Nothing was added to game_log')
            return 0
        if self.ngames==self.batch_size:
            print('Error: trying to add game number',self.ngames+1,'to batch',self.nbatches+1,'[ batch size =',self.batch_size,']')
            return 0

        i0 = self.total_in_batch
        discounted_rewards = my_models.discount_rewards(self.turns_reward,dr_gamma,norm_r)
        running_reward = np.cumsum(self.turns_reward)
        self.running_reward = running_reward[0:self.nturns]

        for i in range(self.nturns):
            self.games_cstate[i0+i,:,:] = self.turns_cstate[i,:,:]
            self.games_action[i0+i] = self.turns_action[i]
            self.games_reward[i0+i] = discounted_rewards[i]
            self.games_running_reward[i0+i] = running_reward[i]

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

    def add_game_performance(self,num_in_batch,loss,cross_entropy=-1):

        if num_in_batch>=self.batch_size or num_in_batch<0:
            print('Error: cannot add NN performance data for game',num_in_batch,'[ >',self.batch_size,']')
            return False

        self.games_loss[num_in_batch]=loss
        if cross_entropy>=0:
            self.games_cross_entropy[num_in_batch]=cross_entropy

        return True

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

        self.batch_ave_loss[self.nbatches] = np.mean(self.games_loss[0:iend])
        self.batch_std_loss[self.nbatches] = np.std(self.games_loss[0:iend])

        self.batch_ave_ce[self.nbatches] = np.mean(self.games_cross_entropy[0:iend])
        self.batch_std_ce[self.nbatches] = np.std(self.games_cross_entropy[0:iend])

        self.nbatches+=1

    def get_training_data(self,ngames=-1):

        i0     = 0
        nturns = int(self.total_in_batch)
        iend   = nturns

        # select ngames most recent games
        if ngames<0:
            pass # negative input defaults to all games in batch
        elif ngames>=0 and ngames<=self.ngames:
            nturns = int(np.sum(self.games_length[self.ngames-ngames:self.ngames]))
            i0 = int(iend - nturns)
        else:
            print('Error: ngames =',ngames,'is more than total_in_batch [ =',self.total_in_batch,']')
            return 0,0,0

        states = np.zeros([nturns,self.nrows,self.ncols,1])
        for i in range(nturns):
            states[i,:,:,0] = self.games_cstate[i0+i,:,:]

        # appropriate dimensions are taken care of in split_board with argument self.batch
        # sep_states = my_models.split_board(self.games_cstate,self.total_in_batch)

        actions = self.games_action[i0:iend]
        rewards = self.games_reward[i0:iend]

        # return raw_states,sep_states,actions,rewards
        return states,actions,rewards


    def get_batch_record(self,fetch_batches=[-1],percentage=True,sum_batches=False):
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
            elif bat<0: indx = self.nbatches+bat
            else:       indx = bat

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

    def get_running_reward(self,all_batch=False):

        if all_batch:
            return self.games_running_reward.copy()
        else: # only for most recent game
            return self.running_reward.copy()

    def get_batch_rewards(self,fetch_batches=[-1],sum_batches=False):
        """ returns reward data
        """

        if hasattr(fetch_batches,"__len__"):
            nfetch = len(fetch_batches)
        else:
            nfetch=1
            fetch_batches = [fetch_batches]

        avg_rew = np.zeros(nfetch)
        std_rew = np.zeros(nfetch)

        for i,bat in enumerate(fetch_batches):
            indx = 0
            if np.abs(bat)>self.total_in_batch+1:
                print('Error: cannot get rewards for batch ',bat,'.. [ >',self.nbatches,']')
                return 0,0
            elif bat<0: indx = self.nbatches+bat
            else:       indx = bat

            avg_rew[i] = self.batch_ave_reward[indx]
            std_rew[i] = self.batch_std_reward[indx]

        if sum_batches: # return a single value for avg and std
            avg_rew = np.mean(avg_rew)
            std_rew = np.mean(std_rew)

        return avg_rew,std_rew

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

        max_in_batch = self.max_turns*self.batch_size
        # pre-declared numpy containers that can contain up to max_batch elements
        self.games_cstate = np.zeros([max_in_batch,self.nrows,self.ncols])
        self.games_nstate = np.zeros([max_in_batch,self.nrows,self.ncols])
        self.games_action = np.zeros(max_in_batch)
        self.games_reward = np.zeros(max_in_batch)
        self.games_running_reward = np.zeros(max_in_batch)
        self.running_reward=0 # easy-to-use variable size container

        self.games_total_reward = np.zeros(self.batch_size)
        self.games_record = np.zeros(self.batch_size)
        self.games_length = np.zeros(self.batch_size)
        self.games_loss = np.zeros(self.batch_size)
        self.games_cross_entropy = np.zeros(self.batch_size)

        self.reset_turn_log()

    def reset_batch_log(self):
        self.nbatches=0
        self.batch_record = np.zeros([self.max_batches,5])
        self.batch_ave_reward = np.zeros(self.max_batches)
        self.batch_std_reward = np.zeros(self.max_batches)
        self.batch_ave_turns  = np.zeros(self.max_batches)
        self.batch_std_turns  = np.zeros(self.max_batches)
        self.batch_ave_loss   = np.zeros(self.max_batches)
        self.batch_std_loss   = np.zeros(self.max_batches)
        self.batch_ave_ce     = np.zeros(self.max_batches)
        self.batch_std_ce     = np.zeros(self.max_batches)
        self.reset_game_log()

    def regroup(self,x,naverage=1):

        if naverage<=1:
            return x
        elif naverage>len(x):
            print('Error: Cannot re-group',len(x),'points into %.0f'%naverage,'points.')
            return x

        new_length = round(len(x)/naverage)
        ave_x = np.zeros([new_length])

        sum_x = 0.0
        j = 0
        for i in range(len(x)):
            sum_x+=x[i]
            if i%naverage==0:
                ave_x[j]=sum_x/float(naverage)
                j+=1
                sum_x=0.0

        return ave_x

    def plot_stats(self,game_name,ngroup=-1):

        # plot variables which coincide with episodes [eps] array
        eps = range(0,self.num_episodes,self.batch_size)
        x = self.regroup(eps,ngroup)

        fig = plt.figure(figsize=(10,8), dpi=90)
        fig.patch.set_facecolor('white')
        fig.suptitle(game_name+' Training', fontsize=20, fontweight='bold')

        ax = fig.add_subplot(221)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Average Reward', fontsize=14)
        y = self.regroup(self.batch_ave_reward,ngroup)
        dy = self.regroup(self.batch_std_reward,ngroup)
        plt.plot(x,y,'k-')
        plt.fill_between(x,y-dy,y+dy,color='b',alpha=0.2)

        ax = fig.add_subplot(222)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Average Loss', fontsize=14)
        y = self.regroup(self.batch_ave_loss,ngroup)
        dy = self.regroup(self.batch_std_loss,ngroup)
        plt.plot(x,y,'k-')
        plt.fill_between(x,y-dy,y+dy,color='r',alpha=0.2)

        ax = fig.add_subplot(223)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Average Turns', fontsize=14)
        y = self.regroup(self.batch_ave_turns,ngroup)
        dy = self.regroup(self.batch_std_turns,ngroup)
        plt.plot(x,y,'k-')
        plt.fill_between(x,y-dy,y+dy,color='g',alpha=0.2)

        ax = fig.add_subplot(224)
        ax.set_xlabel('Number of Games', fontsize=14)
        ax.set_ylabel('Performance per batch', fontsize=14)

        _,won,lost,drew,step = self.get_batch_record(fetch_batches=np.arange(self.nbatches),
                                                            sum_batches=False,percentage=True)
        w = self.regroup(won,ngroup)
        l = self.regroup(lost,ngroup)
        d = self.regroup(drew,ngroup)
        s = self.regroup(step,ngroup)

        plt.plot(x,w,'g-',label='won')
        plt.plot(x,l,'r-',label='lost')
        plt.plot(x,d,'b-',label='drew')
        plt.plot(x,s,'k:',label='out-of-steps')
        plt.legend(fontsize=9)

        # plt.show()
        plt.pause(30)
        plt.savefig('training_'+game_name)
