import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from datetime import datetime
import itertools

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=True, epsilon=0.87, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        # spawning dict for new states and 

        # to store turn
        self.t = 0
        self.state_def = [
            ['left', 'right', 'forward'],       #waypoint
            ['red', 'green'],                   #light
            ['left', 'right', 'forward', None], #vehicleleft
            ['left', 'right', 'forward', None], #vehicleright
            ['left', 'right', 'forward', None]  #vehicleoncoming
        ]


        # output like {'Right': 0.0, 'forward': 0.0, 'Left': 0.0, None: 0.0}
        # In a state, the agent can perform above actions
        # the below code produces a dict , with reward for each action = 0
        # initial rewards for all states in state space model is 0.0

        self.template_q = dict((k, 0.0) for k in self.valid_actions)

        # produces a dict where key is a particular state, value is dict which contains { action : reward} pairs
        # sample output ('forward', 'green', None, 'right', 'right'): {'forward': 0.0, None: 0.0, 'Right': 0.0, 'Left': 0.0}
        # first part consit of a particular state , second part contains a dict of ACTION : REWARD pairs

        # conisering state_def above,  we have 3 * 2 * 4 * 4 * 4 states 
        #  these states act as key, and the value is set of actions possible in a state
        #  with initial reward for every action set to 0{'forward': 0.0, None: 0.0, 'Right': 0.0, 'Left': 0.0}

        for state_tuple in itertools.product(*self.state_def):
            self.Q[state_tuple] = self.template_q.copy()




    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0

        if testing:
            epsilon = 0
            alpha = 0
        else:
            # implemented linear decay function
            # with this safety improved to C , reliability decreased to F
            self.epsilon -= 0.01

            # implementing exponential decay
            # normal exponential function is decaying fast, need slow down decay of exponential function
            #self.epsilon = math.exp(-self.alpha*self.t)
            
            # implementing a cos decay function 
            # using cos decay funcion, reliability improved to B, but safety remained at F
            #self.epsilon = math.cos(self.alpha*self.t)
            self.t += 1

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer eatures outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint, inputs['light'], inputs['left'], inputs['right'], inputs['oncoming'])
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = max(self.Q[state].values())
        maxQ_actions = []
        for action, Q in self.Q[state].items():
            if Q == maxQ:
                maxQ_actions.append(action)

        return maxQ, maxQ_actions


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        if not self.learning:
            # dude, agent not in learning state so return,, 
            return
        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0


        # if the state is not in dictionary Q, then insert that state,

        if not state in self.Q:
            self.Q[state] = self.template_q.copy()
            print "state not in Q"

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        #action = None
        

        # changes i made for Question 1
        #CASE 1, Not Learning,self.learning = false
        # valid_actions will give the set of valid actions
        #random.seed(datetime.now())
        #index = random.randint(0, 3)
        #action = self.valid_actions[index]

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        
        # the if condition satsfies the first 2 conditions
        # if not learning, choose a random action
        # random.random()gives a value in [0,1), each action we take is stochastic
        # when epsilon is more, higher the chance random.random() <= epsilon
        # more probability that the agent will choose a random action


        if not self.learning or random.random() <= self.epsilon:
            action = random.choice(self.valid_actions)
        else:
            maxQ, maxQ_actions = self.get_maxQ(state)
            action = random.choice(maxQ_actions)

        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        # using value iteration

        if self.learning:
            self.Q[state][action] = reward * self.alpha + self.Q[state][action] * (1 - self.alpha)

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run()


if __name__ == '__main__':
    run()
