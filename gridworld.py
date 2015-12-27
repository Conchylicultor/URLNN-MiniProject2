from pylab import *                                                                                                  
import numpy
from time import sleep
import os, shutil

        
class Gridworld:
    """
    A class that implements a quadratic NxN gridworld. 
    
    Methods:
    
    learn(N_trials=100)  : Run 'N_trials' trials. A trial is finished, when the agent reaches the reward location.
    visualize_trial()  : Run a single trial with graphical output.
    reset()            : Make the agent forget everything he has learned.
    plot_Q()           : Plot of the Q-values .
    learning_curve()   : Plot the time it takes the agent to reach the target as a function of trial number. 
    navigation_map()     : Plot the movement direction with the highest Q-value for all positions.
    """    
        
    def __init__(self,N,reward_position=(0.8,0.8),obstacle=False, lambda_eligibility=0.95):
        """
        Creates a quadratic NxN gridworld. 

        Mandatory argument:
        N: size of the gridworld

        Optional arguments:
        reward_position = (x_coordinate,y_coordinate): the reward location
        obstacle = True:  Add a wall to the gridworld.
        """    
        
        
        print 'Warning: Erase all previous results in \'results/\''
        folder = 'results/'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception, e:
                print e
        
        # gridworld size
        self.N = N

        # reward location
        self.reward_position = reward_position        

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 10
        self.reward_at_wall   = -2

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        self.epsilon = 0.5

        # learning rate
        self.eta = 0.005

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = 0.95

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility
    
        # is there an obstacle in the room?
        self.obstacle = obstacle

        # length of the steps the rat makes
        self.step_length = 0.03
        
        self._isRecording = True # useless (not used)
        self._isVerbose = False
        self._isVisualization = False
        self._isStopped = False

        # initialize the Q-values etc.
        self._init_run()

    def run(self,N_trials=50,N_runs=10):
        self.latencies = zeros(N_trials)
        
        for run in range(N_runs):
            self._init_run()
            # Call reset() to reset Q-values and latencies, ie forget all he learnt
            self.reset() # All runs are independant
            
            """
            Run a learning period consisting of N_trials trials. 
            
            Options:
            N_trials :     Number of trials

            Note: The Q-values are not reset. Therefore, running this routine
            several times will continue the learning process. If you want to run
            a completely new simulation, call reset() before running it.
            
            """
            for trial in range(N_trials):
                print 'Start trial ', run, ', ', trial
                # run a trial and store the time it takes to the target
                latency = self._run_trial()
                
                self.latency_list.append(latency)
                imsave('results/' + str(run) + '_' + str(trial) + '.png', self._display)
                
                self.navigation_map() # TODO: Check the function
                savefig('results/' + str(run) + '_' + str(trial) + '_navigationMap_.png')
                
                if self._isVisualization:
                    self._close_visualization()
                
                print 'Results saved'

            points = [(0.1,0.1),(0.1,0.9),(0.5,0.1),(0.5,0.9),(0.9,0.1)]
            for k in points:
                print 'best action to take at ', k , ' is ', argmax(self.compute_Q(k[0],k[1],range(8)))

            latencies = array(self.latency_list)
            self.latencies += latencies/N_runs
            
            self.learning_curve() # TODO: Check the function
            savefig('results/' + str(run) + '_learningCurve_.png')

    # def visualize_trial(self):
    #     """
    #     Run a single trial with a graphical display that shows in
    #             red   - the position of the agent
    #             blue  - walls/obstacles
    #             green - the reward position

    #     Note that for the simulation, exploration is reduced -> self.epsilon=0.1
    
    #     """
    #     # store the old exploration/exploitation parameter
    #     epsilon = self.epsilon

    #     # favor exploitation, i.e. use the action with the
    #     # highest Q-value most of the time
    #     self.epsilon = 0.1

    #     self._run_trial(visualize=True)

    #     # restore the old exploration/exploitation factor
    #     self.epsilon = epsilon

    def learning_curve(self,log=False,filter=1.):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Options:
        filter=1. : timescale of the running average.
        log    : Logarithmic y axis.
        """
        figure(3) #a matplotlib figure instance
        xlabel('trials')
        ylabel('time to reach target')
        latencies = array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter'
        for i in range(1,latencies.shape[0]):
            latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter)

        if not log:
            plot(self.latencies)
        else:
            semilogy(self.latencies)

    def navigation_map(self):
        """
        Plot the direction with the highest Q-value for every position.
        Useful only for small gridworlds, otherwise the plot becomes messy.
        """
        self.x_direction = numpy.zeros((self.N,self.N))
        self.y_direction = numpy.zeros((self.N,self.N))

        self.actions = argmax(self.W[:,:,:],axis=2)
        self.y_direction[self.actions==0] = 1 # Up
        self.y_direction[self.actions==1] = 1 # Up right
        self.y_direction[self.actions==2] = 0 # Right
        self.y_direction[self.actions==3] = -1 # Down right
        self.y_direction[self.actions==4] = -1 # Down
        self.y_direction[self.actions==5] = -1 # Down left
        self.y_direction[self.actions==6] = 0 # Left
        self.y_direction[self.actions==7] = 1 # Up left
        
        self.x_direction[self.actions==0] = 0
        self.x_direction[self.actions==1] = 1
        self.x_direction[self.actions==2] = 1
        self.x_direction[self.actions==3] = 1
        self.x_direction[self.actions==4] = 0
        self.x_direction[self.actions==5] = -1
        self.x_direction[self.actions==6] = -1
        self.x_direction[self.actions==7] = -1

        figure(2)
        clf()
        quiver(self.x_direction,self.y_direction)
        axis([-0.5, self.N - 0.5, -0.5, self.N - 0.5])

    def reset(self):
        """
        Reset the W-values (and the latency_list).
        
        Instant amnesia -  the agent forgets everything he has learned before    
        """
        self.W = numpy.random.rand(self.N,self.N,8)
        self.latency_list = []

    def plot_Q(self):
        """
        Plot the dependence of the Q-values on position.
        The figure consists of 4 subgraphs, each of which shows the Q-values 
        colorcoded for one of the actions.
        """
        figure()
        for i in range(4):
            subplot(2,2,i+1)
            imshow(self.W[:,:,i],interpolation='nearest',origin='lower',vmax=1.1)
            if i==0:
                title('Up')
            elif i==1:
                title('Down')
            elif i==2:
                title('Right')
            else:
                title('Left')

            colorbar()
        draw()

    ###############################################################################################
    # The remainder of methods is for internal use and only relevant to those of you
    # that are interested in the implementation details
    ###############################################################################################


    def _init_run(self):
        """
        Initialize the W-values, eligibility trace, position etc.
        """
        # initialize the W-values and the eligibility trace
        self.W = 0.01 * numpy.random.rand(self.N,self.N,8) + 0.1
        self.e = numpy.zeros((self.N,self.N,8))
        
        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.x_position = None
        self.y_position = None
        self.action = None

    def _run_trial(self):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """
        # choose the initial position (0.1,0.1)
        self.x_position = 0.1
        self.y_position = 0.1
        
        maxIter = 10000
                
        print "Starting trial at position ({0},{1}), reward at ({2},{3})".format(self.x_position,self.y_position,self.reward_position[0],self.reward_position[1])
        if self.obstacle:
            print "Obstacle is in position (?,?)"

        # initialize the latency (time to reach the target) for this trial
        latency = 0.

        # start the visualization, if asked for
        self._init_visualization()
            
        # run the trial
        self._choose_action()
        while not self._arrived():
            self._update_state()
            self._choose_action()    
            self._update_w()

            self._visualize_current_state(latency)
        
            latency = latency + 1
            if latency > maxIter:
                break;

        if self._arrived():
            print 'Reward reached in ', latency, ' steps'
        else:
            print 'Aborted'
            latency = -1 # TODO
            
        return latency

    def _update_w(self):
        """
        I changed this from _update_Q to _update_w because for each state that
        has a non-zero eligibility trace, we must update the weights associated
        with the action taken at that state for all centres j - but I am not
        sure how r_j is used in this calculation?
        """
        # update the eligibility trace
        self.e = self.gamma * self.lambda_eligibility * self.e # TODO: Is it the right gamma ?TODO: Where is gamma ? e(t+1)=gamma*lambda*e(t) + ...
        for i in range(self.N):
            for j in range(self.N):
                self.e[i,j,self.action] += self.compute_rj(self.x_position,self.y_position,i,j) # TODO: Old position ?? Or new one ?? << In the original code, it was the old position used to update e
                # (I may be wrong but I think the old position correspond to the current position in the sarsa algorithm)
                # updated based on action taken in state (x,y).
                #print i, '-', j, ': ', self.compute_rj(self.x_position_old, self.y_position_old, i, j) # TODO: Too big. Pb with sigma in the gaussian ?

        # update the weights
        if self.action_old != None:
            q_old = self.compute_Q(self.x_position_old,self.y_position_old,self.action_old)
            q_new = self.compute_Q(self.x_position, self.y_position, self.action)
            delta_t = self._reward() - (q_old - self.gamma*q_new)
            self.W += self.eta * delta_t * self.e

    def _choose_action(self):    
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action 
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        if numpy.random.rand() < self.epsilon:
            self.action = numpy.random.randint(8)
            # print 'Randomly pick action', self.action
        else:
            Q_values = numpy.zeros(8) # 1 Q-value per action
            for i_action in range(8):
                Q_values[i_action] = self.compute_Q(self.x_position, self.y_position, i_action)
            self.action = argmax(Q_values)

            #print Q_values
            #print self.action # TODO: Strange, action plot smaller and smaller values (bug in update rule?)
            
            # print 'best action picked:', self.action

    def compute_Q(self, x_pos, y_pos, i_action):
        Q_value = 0
        for i in range(self.N):
            for j in range(self.N):
                Q_value += self.W[i,j,i_action] * self.compute_rj(x_pos, y_pos, i, j)
        return Q_value

    def compute_rj(self, x_pos, y_pos, i_val, j_val):
        sigma = 0.05
        xj = i_val/(self.N-1.)
        yj = j_val/(self.N-1.)
        rj = exp(-((xj-x_pos)**2 + (yj - y_pos)**2)/(2*(sigma**2)))
        return rj

    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        return (self.x_position - self.reward_position[0])**2 +\
               (self.y_position - self.reward_position[1])**2 <= 0.1**2

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the 
        chosen action at the current location
        """
        if self._arrived():
            return self.reward_at_target

        if self._wall_touch:
            return self.reward_at_wall
        else:
            return 0.

    def _update_state(self):
        """
        Update the state according to the old state and the current action.    
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        step_length = self.step_length
        # update the agents position according to the action
        #  move up
        if self.action == 0:
            self.y_position += step_length
        # move up right
        elif self.action == 1:
            self.x_position += sqrt((step_length**2)/2)
            self.y_position += sqrt((step_length**2)/2)
        # move right
        elif self.action == 2:
            self.x_position += step_length
        # move down right
        elif self.action == 3:
            self.x_position += sqrt((step_length**2)/2)
            self.y_position -= sqrt((step_length**2)/2)
        #  move down
        elif self.action == 4:
            self.y_position -= step_length
        #  move down left
        elif self.action == 5:
            self.y_position -= sqrt((step_length**2)/2)
            self.x_position -= sqrt((step_length**2)/2)
        #  move left
        elif self.action == 6:
            self.x_position -= step_length
        #  move up left
        elif self.action == 7:
            self.x_position -= sqrt((step_length**2)/2)
            self.y_position += sqrt((step_length**2)/2)
        else:
            print "There must be a bug. This is not a valid action!"
        
        if self._isVerbose:
            print "({0},{1}) >> ({2},{3})".format(self.x_position_old,self.y_position_old,self.x_position,self.y_position)
        
        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
            if self._isVerbose:
                print "#### wally ####"
        else:
            self._wall_touch = False

    def _is_wall(self,x_position=None,y_position=None):    
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have 
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position == None or y_position == None:
            x_position = self.x_position
            y_position = self.y_position

        # check of the agent is trying to leave the gridworld
        if x_position <= 0 or x_position >= 1 or y_position <= 0 or y_position >= 1:
            return True

        # check if the agent has bumped into an obstacle in the room
        if self.obstacle: # TODO: No obstacle (never ?) < To remove ?
            if y_position == 1/2 and x_position>1/2:
                return True

        # if none of the above is the case, this position is not a wall
        return False 
            
    def _visualize_current_state(self, latency):
        """
        Show the gridworld. The squares are colored in 
        red - the position of the agent - turns yellow when reaching the target or running into a wall
        blue - walls
        green - reward
        """

        # set the agents color
        self._update_display(self.x_position_old, self.y_position_old, 0, 0.1+abs(sin(latency/70.)/2.)*0.8) # Decrease color over time
        self._update_display(self.x_position_old, self.y_position_old, 2, 0.1+abs(sin(latency/1000.)/2.)*0.8) # Decrease color over time
        # self._update_display(self.x_position_old, self.y_position_old, 0, 0.5) # Cst color
        self._update_display(self.x_position_old, self.y_position_old, 1, 0)
        self._update_display(self.x_position, self.y_position, 0, 1)

        if self._wall_touch:
            self._update_display(self.x_position, self.y_position, 1, 1)
            
        # set the reward locations
        self._update_display(self.reward_position[0], self.reward_position[1], 1, 1)

        # update the figure
        if self._isVisualization:
            figure(1)
            self._visualization.set_data(self._display)
            draw()
        
        # and wait a little while to control the speed of the presentation
        sleep(0.01)
        
    def _init_visualization(self):
        
        # create the figure
        figure(1)
        # initialize the content of the figure (RGB at each position)
        self._display = numpy.zeros((self.N,self.N,3))

        # position of the agent
        self._update_display(self.x_position, self.y_position, 0, 1)
        self._update_display(self.reward_position[0], self.reward_position[1], 1, 1)
                
        #for x in range(self.N): # TODO: No obstacles ?
        #    for y in range(self.N):
        #        if self._is_wall(x_position=x/19,y_position=y/19):
        #            self._display[x/19,y/19,2] = 1.

        self._visualization = imshow(self._display,interpolation='nearest',origin='lower')
        
        if self._isVisualization:
            ion()
            show()
        
    def _update_display(self, x, y, channel, value):
        """
        Convert continious position to discrete for ploting (from [0,1] to [0,N])
        """
        self._display[floor(x*self.N), floor(y*self.N), channel] = value

    def _close_visualization(self):
        if self._isStopped:
            print "Press <return> to proceed..."
            raw_input()
        figure(1)
        close()
        figure(2)
        close()

if __name__ == "__main__":
    grid = Gridworld(20)
    #grid.run(50, 10); # For the final run
    grid.run(150, 3);
