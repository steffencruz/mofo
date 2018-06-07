"""
More-Four game environment, made by Steffen Cruz 2018

A twist on the classic connect-4 game, where connecting 4 counters
in a straight line scores a point and earns the player another turn.
With a strong playing strategy, a player can dominate the board
their opponent by accumulating rewards and additional turns.

The winner is the player with the most points when the board is full.

This can be played with any board size or shape,
which significantly affects the complexity of the game and possible strategies.

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

"""

import math
import gym
from gym import error, spaces, logger, utils
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
import tensorflow as tf
import time

# import my_models

class MoFoEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,nrows=7,ncols=7,verbose=0,training=True,testing=False):

        """
        initializes core variables and begins game
        Args: number of columns and rows
        """
        # control verbosity
        self.verbose = verbose
        # control rules and game flow
        self.Training = training
        self.Testing  = testing

        self.NCols = ncols
        self.NRows = nrows
        # create the board and fill with zeros
        # 1 (2) - if a cell is occupied by player 1 (2)
        self.Board = np.zeros([nrows,ncols],dtype=int)
        self.Length = 4 # condition to score a strike
        self.Points = [0,0]
        self.Scored = 0
        self.Turn = 0

        # marker size
        self.MSize = 350/np.max([self.NCols,self.NRows])

        # two masks which shows only strikes for each player
        self.Strike = np.zeros([2,nrows,ncols])

        # history of moves played [player x y]
        self.MovesPlayed = np.zeros([1,5],dtype=int)
        self.StrikesPlayed = np.zeros([1,7],dtype=int)

        # Player describes the game state in the next round
        # 0 - game over
        # 1 - player 1 turn
        # 2 - player 2 turn
        self.Player = 1
        self.Valid_Turn = True

        self.AI = False
        self.AI_input = None
        self.AI_action = None

        self.action_space = spaces.Discrete(ncols)
        self.observation_space = spaces.Box(-1,-1,[nrows,ncols],dtype='int')

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def initialize(self,nrows=4,ncols=4,verbose=0,training=True,testing=False):

        """
        initializes core variables and begins game
        Args: number of columns and rows
        """
        self.__init__(nrows,ncols,verbose,training,testing)

    def reset(self):

        """
        Resets state of game
        """

        self.steps_beyond_done = None

        self.Board = np.zeros([self.NRows,self.NCols],dtype=int)
        self.Points = [0,0]
        self.Turn = 0
        self.Scored = 0

        self.Strike = np.zeros([2,self.NRows,self.NCols])
        self.MovesPlayed = np.zeros([1,5],dtype=int)
        self.StrikesPlayed = np.zeros([1,7],dtype=int)

        self.Player = 1
        self.Valid_Turn = True

        self.state = self.Board

        self.AI_input = None
        self.AI_action = None

        self.seed()

        self.close()
        self.viewer = None
        # if self.viewer: self.viewer.window.clear()# = None

        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        """
        takes care of who's turn it is and when the game ends
        In training mode, this function only returns when it's player 1's turn to choose and action or game over
        In normal mode, this function returns after every move
        """

        # first make sure that the action is in range
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        p = self.Player
        self.Turn+=1
        reward = 0

        # if self.verbose>0: print('\nTurn',self.Turn,': Player',p,', go!')

        if self.Training or self.Testing:
            # use action
            x = action
            # determine reward, check who's turn is next and if game is over
            reward_p1 = self.PlayTurn(x,p)
            # reward_p1 = np.max([0,reward_p1])
            p = self.Player
            if p==0:  # game over
                return self.Board.copy(), reward_p1, True, self.Winner
            elif p==1: # p1 choose new action
                return self.Board.copy(), reward_p1, False, {}
            while p==2: # automatically choose p2 action and update until p1 turn
                self.Turn+=1
                x = self.AITurn()
                # determine p2 reward [unused], check who's turn is next and if game is over
                reward_p2 = self.PlayTurn(x,p)
                p = self.Player

                # if opponent scores then subtract this from reward
                if reward_p2>0:
                    reward_p1 -= reward_p2

            if p==0:  # game over
                return self.Board.copy(), reward_p1, True, self.Winner
            elif p==1: # p1 choose new action with new board state
                return self.Board.copy(), reward_p1, False, {}
        else:
            # apply user action each turn
            x = action
            # determine reward, check who's turn is next and if game is over
            reward = self.PlayTurn(x,p)
            p = self.Player
            if p==0:  # game over
                return self.Board.copy(), reward, True, self.Winner
            else: # return control to next player
                return self.Board.copy(), reward, False, p

    def PlayGame(self,is_human=[True,False],model_dir=''):
        if np.all(is_human)==False:
            self.InitAI(model_dir)

        print('\n\n__________ MoFo ___________')
        print('Player 1 is human:',is_human[0])
        print('Player 2 is human:',is_human[1])

        self.Training = False
        self.verbose=1
        p = self.Player
        while p!=0:

            if is_human[p-1]:
                self.PrintBoard(fancy=True)
                istr = "Human player "+str(p)+", choose a column number: "
                while True:
                    x = input(istr)
                    if len(x)>0:
                        x=int(x)
                        break
            else:
                x = self.AITurn()

            self.PlayTurn(x,p)
            p = self.Player

        return self.Points

    def InitAI(self,import_dir,filename=''):

        if len(filename)==0:
            # if no filename given search for latest model that was saved
            filename = tf.train.latest_checkpoint(import_dir+filename)
            saver = tf.train.import_meta_graph(filename+'.meta')
        else:
            saver = tf.train.import_meta_graph(import_dir+'/'+filename+'.meta')

        self.sess = tf.Session()
        saver.restore(self.sess,filename)
        self.graph = tf.get_default_graph()
        self.AI = True

        print('Loaded most recent AI:',filename)
        return True

    def AITurn(self):
        """ computer opponents are trained RL agent or random number generator"""

        if self.AI:
            self.AI_input  = self.graph.get_collection('input_layer')[0]
            self.AI_action = self.graph.get_collection('sample_op')[0]
            # action = self.sess.run(self.AI_action,{self.AI_input:my_models.split_board(self.Board)})
            action = self.sess.run(self.AI_action,{
                        self.AI_input:self.Board.reshape((1,self.NRows,self.NCols,1))})
            return action[0]
        else:
            return np.random.randint(self.NCols)

    def render(self, mode='human'):

        """
        Draws the current state of the game
        """

        if self.viewer is None:
            screen_width = 800
            screen_height = 600
            self.viewer = rendering.Viewer(screen_width, screen_height)

        if self.state is None: return None

        # rescale board
        a = 60 # width
        b = 60 # height

        turns = np.where(self.MovesPlayed[:,4]>=0)[0] # loop over valid moves
        for i in turns:

            if mode=='human': # to draw turn-by-turn, skip to last counter
                i = turns[-1]

            p = self.MovesPlayed[i,1] # player
            x = self.MovesPlayed[i,2] # x
            y = self.MovesPlayed[i,3] # y
            s = self.MovesPlayed[i,4] # score

            # doing this every time (not calling same object each time) should add an additional counter
            counter = rendering.make_circle(18)
            if p==1:
                counter.set_color(.8,.1,.2)
            elif p==2:
                counter.set_color(.1,.8,.2)

            self.counter_trans = rendering.Transform()
            counter.add_attr(self.counter_trans)
            self.counter_trans.set_translation(a+b*x,a+b*y)

            # updating the position of the same counter 'glow' object will only produce one on screen
            if i == turns[-1]:
                self.new_counter = rendering.make_circle(22)#self.MSize*1.2)
                self.new_counter.set_color(1,1,0)
                self.new_counter.add_attr(self.counter_trans)
                self.viewer.add_geom(self.new_counter)    # this might produce many copies

            # print('Added counter. Turn',i,'Player',p)
            self.viewer.add_geom(counter)


        if mode=='human' and self.Scored>0: # draw line if a point is scored
            for i in range(self.Scored):
                ys = self.StrikesPlayed[len(self.StrikesPlayed)-i,2:6:2]
                xs = self.StrikesPlayed[len(self.StrikesPlayed)-i,3:6:2]

                # draw strike as a polyline
                self.strike = rendering.make_polyline(list(zip(a+b*xs,a+b*ys)))
                self.strike.set_linewidth(15.0)

                if p==1:
                    self.strike.set_color(.7,.1,.3)
                elif p==2:
                    self.strike.set_color(.1,.7,.3)

                self.viewer.add_geom(self.strike)
        else:
            for i in range(len(self.StrikesPlayed)):

                p = self.StrikesPlayed[i,1] # player
                ys = self.StrikesPlayed[i,2:6:2]
                xs = self.StrikesPlayed[i,3:6:2]

                # draw strike as a polyline
                self.strike = rendering.make_polyline(list(zip(a+b*xs,a+b*ys)))
                self.strike.set_linewidth(15.0)

                if p==1:
                    self.strike.set_color(.7,.1,.3)
                elif p==2:
                    self.strike.set_color(.1,.7,.3)

                self.viewer.add_geom(self.strike)

        # txt = '- Player 1:  {} -'.format(self.Points[0])
        # self.txt1.set_text(txt)
        # txt = '- Player 2:  {} -'.format(self.Points[1])
        # self.txt2.set_text(txt)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

    def PlayTurn(self,x,p):

        msg=''
        if self.Training and p==1: msg = 'with purpose'
        if self.Training and p==2: msg = 'randomly'

        # locate the next unnoccupied rows [y] in column x
        ytmp = np.where(self.Board[:,x]==0)[0]

        if len(ytmp)<1: y = -1 # flag for invalid move
        else:           y = ytmp[0] # place counter in first available row
        # could also make a variation of game that randomizes y position

        if self.verbose>0: print('Turn',self.Turn,'--> Player',p,'chooses x =',x,msg,'y =',ytmp)

        # look for invalid move or any new strikes
        reward = self.CheckState(x,y)

        if self.verbose>1: self.PrintBoard()

        # switch players or end game, as necessary
        nturns_rem = self.CountCells(self.Board)
        done = False
        if nturns_rem==0: # game over
            self.Player = 0 # set Player to game over
            done = True
            if self.Points[0]==self.Points[1]: # draw scenario
                self.Winner = 0
                self.Score = np.max(self.Points)
                if self.verbose>0: print("\n\tGAME OVER. Scores",self.Points,"It's a draw with",self.Score,"points !\n")
            else:
                self.Winner = np.argmax(self.Points)+1
                self.Score = np.max(self.Points)
                if self.verbose>0: print("\n\tGAME OVER. Scores",self.Points,"Winner is player",self.Winner,"with",self.Score,"points !\n")
        # if you make an invalid move you retake turn
        # if valid but no score, switch players next turn..otherwise same player
        # elif reward<1: self.Player = p%2 + 1
        elif reward==0 and self.Valid_Turn: self.Player = p%2 + 1 # don't change players if reward<0

        if done and self.steps_beyond_done is None: # game just ended
            self.steps_beyond_done = 0
        elif done: # prints warning if game continues
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            print('steps_beyond_done=',self.steps_beyond_done)
            self.steps_beyond_done += 1
            reward = 0.0

        if self.verbose>1: print('. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n')

        return reward

    def CheckState(self,x,y):

        """
        checks whether move was valid, and updates scores
        looks for all new possible matches along the 4 intersecting lines
        """

        if self.CountCells(self.Board,0)-self.NRows*self.NCols>=2*self.Length:
            return 0

        score = 0
        if y<0:
            self.Valid_Turn = False
            if self.Training:
                # invalid move [y=-1] receives a -1 point penalty in training
                score = -1
        else:
            self.Valid_Turn = True
            # update board if move was valid
            self.Board[y,x] = self.Player

            # now check for any strikes and add scored points
            xx = np.arange(x-self.Length+1,x+self.Length)
            yy = np.arange(y-self.Length+1,y+self.Length)
            yy2 = np.arange(y+self.Length-1,y-self.Length,-1)

            score  = self.FindStrike(xx,np.ones(np.shape(xx),dtype=int)*y,0)
            score  += self.FindStrike(np.ones(np.shape(yy),dtype=int)*x,yy,1)
            score  += self.FindStrike(xx,yy,2)
            score  += self.FindStrike(xx,yy2,3)

        # update player score
        self.Points[self.Player-1] += score

        # some printing
        if score>0:
            if self.verbose>2: self.PrintMoves()
            if self.verbose>1: self.PrintStrikes()
            if self.verbose>0: print("Player",self.Player,"scored",score,", total points =",self.Points[self.Player-1])
        elif self.Valid_Turn==False and self.verbose>0:
            print('Player',self.Player,'made an invalid move, x =',x,', Reward = ',score,', total points =',self.Points[self.Player-1])

        self.Scored = score

        # save this move, whether invalid or valid
        if self.MovesPlayed[0,0]==0: # first turn, initalize
            self.MovesPlayed[0,:] = [self.Turn,self.Player,x,y,score]
        else:
            self.MovesPlayed = np.vstack((self.MovesPlayed,[self.Turn,self.Player,x,y,score]))

        return score

    def FindStrike(self,xx,yy,n):

        L = int(self.Length-1)
        b = np.zeros([self.NRows+2*L,self.NCols+2*L],dtype=int)

        # make a larger board (so that no sequences go out of bounds)
        b[L:L+self.NRows,L:L+self.NCols] = self.Board

        prev_strikes = self.StrikesPlayed[self.StrikesPlayed[:,1]==self.Player,:]

        seq = b[yy+L,xx+L]

        if self.verbose>2:
            print('xx=',xx,'yy=',yy,'seq=',seq,'n=',n)#,'sseq=',sseq)

        score = 0
        for i in range(len(seq)-L):
            # print("\tsubsequence",i,"=",seq[i:i+self.Length])

            # create a subsequence of length 4 to test for a match
            indx = np.arange(i,i+self.Length)
            if all(seq[indx]==self.Player): # if all 4 counters belong to the same player
                # found 4 in a row

                prev_strikes = self.StrikesPlayed[self.StrikesPlayed[:,1]==self.Player,:]
                prev_strikes = prev_strikes[prev_strikes[:,6]==n,:]

                strikes_tmp = self.Strike[self.Player-1,:,:]

                # print('\n\nxx=',xx[indx],'yy=',yy[indx],'strikes_tmp=',strikes_tmp[yy[indx],xx[indx]])

                for j in indx:
                    strikes_tmp[yy[j],xx[j]] += 1

               # currently sets every marker to 1
                self.Strike[self.Player-1,yy[indx] ,xx[indx]] = 1
                valid_strike = True
                # print('\n\nLooping over all previous strikes and comparing to new one..\n')
                for j in range(len(prev_strikes)):
                    yprev = np.linspace(prev_strikes[j,2],prev_strikes[j,4],4,dtype='int')
                    xprev = np.linspace(prev_strikes[j,3],prev_strikes[j,5],4,dtype='int')
                    npts=0
                    # print('\nj=',j,'\nxprev=',xprev,'\t\txcurr=',xx[indx],'\nyprev=',yprev,'\t\tycurr=',yy[indx])
                    for k in range(4):
                        for l in range(i,i+4):
                            if xprev[k]==xx[l] and yprev[k]==yy[l]:
                                npts+=1
                                if npts>1: # if this new strike shares more than 1 point with a previous strike it is not allowed
                                    # print('There are',npts,'shared points. This is too many. INVALID STRIKE.')
                                    valid_strike=False
                                    break
                        if valid_strike==False:
                            break

                if valid_strike==False:
                    # print('\n\nPlayer',self.Player,'DOES NOT score with [',xx[i],yy[i],']-[',xx[i+L],yy[i+L],']')
                    # print('. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n')
                    continue

                    # print('\nj=',j,' Do [',prev_strikes[j,2],prev_strikes[j,3],']-[',prev_strikes[j,4],prev_strikes[j,5],']'
                        # '   and   [',yy[i],xx[i],']-[',yy[i+L],xx[i+L],'] overlap?\n')

                score+=1

                # add strike to list
                if self.StrikesPlayed[0,0]==0:
                    self.StrikesPlayed[0,:] = [self.Turn,self.Player,yy[i],xx[i],yy[i+L],xx[i+L],n]
                else :
                    self.StrikesPlayed = np.vstack((self.StrikesPlayed,[self.Turn,self.Player,yy[i],xx[i],yy[i+L],xx[i+L],n]))

                if self.verbose>0:
                    print('** Player',self.Player,'scores with [',xx[i],yy[i],']-[',xx[i+L],yy[i+L],'] **')

                # we can step forward 3 times as there cannot be strikes before
                i += L

        return score

    def CountCells(self,mat,val=0):

        """
        finds the number of matching elements in a matrix
        Args: matrix and element value
        Returns: number of matching elements
        """

        pos = np.where(mat==val)
        return len(pos[0])
        # return np.max([len(pos[0]),len(pos[1])])

    def PrintBoard(self,fancy=False):

        """
        prints board state
        """

        # vectorize MovesPlayed to keep a history of the game
        x,y = self.MovesPlayed[-1,1:3]

        print('\n\tBOARD')
        if fancy:
            syms = ['  .','  x','  o']
            for j in range(self.NRows-1,-1,-1):
                symstr = '\t'
                # symstr[0] = ' '
                for i in range(self.NCols):
                    symstr += syms[int(self.Board[j,i])]
                print(*symstr,"   ",j,sep="")
            print("\n\t",*range(self.NCols),sep="  ")
            print("\t   column\n")
        else:
            print(np.flipud(self.Board))

        print('\n\tSCORE')
        print('\tPlayer 1 :',self.Points[0],'\tPlayer 2 :',self.Points[1])
        print("\nThere are",self.CountCells(self.Board),"(total) turns remaining..\n")

    def PrintMoves(self):

        syms = ['  .','  x','  o']

        print('\n\tMOVES PLAYED:\n\nTurn\tPlayer\tX\tY\tReward\tMark')
        for i in range(len(self.MovesPlayed)):
            print(*self.MovesPlayed[i,:],syms[self.MovesPlayed[i,1]],sep='\t')

    def PrintStrikes(self):

        cnt = [0,0]
        print('\n\tSTRIKES PLAYED:\n\nTurn\tPlayer\tx1\ty1\tx2\ty2\tSeq\tScore')
        for i in range(len(self.StrikesPlayed)):
            cnt[self.StrikesPlayed[i,1]-1]+=1
            print(*self.StrikesPlayed[i,:],cnt[self.StrikesPlayed[i,1]-1],sep='\t')
        # input('\nPress any key:')
