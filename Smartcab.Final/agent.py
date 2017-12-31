import random
import math
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import code

class LearningAgent(Agent):

    def __init__(self, env, learning=True, epsilon=1.0, alpha=0.5):
    # for question 6 ~ def __init__(self, env, learning=False?, epsilon=1, alpha=0.5):
        super(LearningAgent, self).__init__(env)     
        self.planner = RoutePlanner(self.env, self)  
        self.valid_actions = self.env.valid_actions  

        self.learning = learning 
        self.Q = dict()          
        self.epsilon = epsilon   
        self.alpha = alpha       
        self.trial = 1

        self.valid_actions = [None, 'left', 'right', 'forward']
        
    def reset(self, destination=None, testing=False):

        self.planner.route_to(destination)
      
        if testing:
        	self.epsilon = 0
        	self.alpha = 0
        else:
            #self.tolerance = 0.0005
            t = self.trial
            #self.alpha = .90
            a = self.alpha
            #self.epsilon =  .99**t
            self.trial += 1
            
            # for question 6
            self.alpha = .5
            self.tolerance = 0.05
            self.epsilon =  self.epsilon - .05
           
        return None

    def build_state(self):
   
        waypoint = self.planner.next_waypoint() 
        inputs = self.env.sense(self)           
        deadline = self.env.get_deadline(self)  
        state = (waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        
        return state

    def createQ(self, state):
       
        if self.learning:
            if not self.Q.has_key(state):
                Q_table = dict()
                for action in self.valid_actions:
                    Q_table[action] = 0.0
                self.Q[state] = Q_table
                
        return

    def choose_action(self, state):
      
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None
        
        if self.learning and (random.random() > self.epsilon):
            
            max_Q = None
            Q_table = self.Q[state]
            best_value = -10000
            best_action = None
            action_list = Q_table.keys()
            random.shuffle(action_list)
            for action in action_list:
                if Q_table[action] > best_value:
                    best_value = Q_table[action]
                    best_action = action
            
                action = best_action

        else:
            action = random.choice(Environment.valid_actions)
        
        return action

    def learn(self, state, action, reward):
     
        self.Q[state][action] = (self.Q[state][action] * (1- self.alpha)) + (reward * self.alpha)
        
        return


    def update(self):
  
        state = self.build_state()          
        self.createQ(state)                 
        action = self.choose_action(state)  
        reward = self.env.act(self, action) 
        if self.learning:
            self.learn(state, action, reward)   
        
        return
        

def run():

    env = Environment()

    agent = env.create_agent(LearningAgent, learning=True)
    
    env.set_primary_agent(agent, enforce_deadline = True)

    sim = Simulator(env, log_metrics = True, update_delay=.01, optimized = False, display=False)
     
    #sim.run(n_test=200, tolerance = 0.0005)  # run for a specified number of tests
    # for question six
    sim.run(n_test=10, tolerance = 0.05)  # run for a specified number of tests

   

if __name__ == '__main__':
    run()