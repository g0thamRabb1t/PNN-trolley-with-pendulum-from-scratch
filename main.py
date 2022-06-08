# Important 
# If at some point the program gets stuck, you can pause it in colab, 
# because the best result data will still be saved in last_good_configuration()
# method so you can still plot it with the last lines from the code - just copy 
# the last 20 lines to the next section, stop the execution of that piece of 
# code and launch the next one

import numpy as np
import matplotlib.pyplot as plt
import math
import signal
from contextlib import contextmanager

# A class that allows us to check the execution time of a method, and if takes 
# too long, throws an error - useful for predict(), where sometimes there were 
# numbers so large that python crashed computationally
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# here we will hold the best result and then make a graph of it
last_good_configuration = dict.fromkeys(['time', 'sigma', 'centers', 'weights', 'x_plot', 'y_plot', 'y2_plot', 'y3_plot'])

stick_length = 0.5
stick_length1 = 0.05
stick_weight = 0.1
stick_weight1 = 0.1
trolley_weight = 1
gravitation = 9.8	
miC = 0.0005
miP = 0.000002
time = 0.02

# Here's the code with the calculation for the trolley. The comments below 
# describe what is important in the learning process.
class Pendulum:
    def __init__ (self):
        self.position = 0
        self.angle = 4
        self.angle1 = 0
	
        self.pX = 0
        self.pangle = 0
        self.pangle1 = 0
        self.time = 0

    def get_position(self):
        return self.position

    def get_time(self):
        return self.time
    
    def get_angle(self):
        return self.angle

    def get_angle1(self):
        return self.angle1

    # here is the run() method, which takes as the argument force, which 
		# pushes the trolley with some force at a given moment depending on the 
		# iteration - that is, we will manipulate the force and set the trolley in 
		# motion with the calculated force:
	  # if force = 0 - nothing changes in the motion of the trolley - it goes 
		# according to the speed given in the previous iteration
    # if force <0 - the trolley slows down if it was going forward before or 
		# accelerates backing up if it was going backwards
    # if force> 0 - the trolley accelerates, if it was going forward before
    # or slows down in reverse if it was driving in reverse
    def run(self, force):
        md = stick_weight*stick_length
        md1 = stick_weight1*stick_length1
        
        pom11 = (miP * self.pangle) / md
        pom21 = gravitation*np.sin(self.angle)
        
        pom12 = (miP*self.pangle1)/md1
        pom22 = gravitation*np.sin(self.angle1)
        
        fF = md*self.angle*self.angle*np.sin(self.angle)+0.75*stick_weight*np.cos(self.angle)*(pom11+pom21)
        mF = stick_weight*(1 - 0.75*np.cos(self.angle)*np.cos(self.angle))
        
        fF1 = md1*self.angle1*self.angle1*np.sin(self.angle1)+0.75*stick_weight1*np.cos(self.angle1)*(pom12+pom22)
        mF1 = stick_weight1*(1 - 0.75*np.cos(self.angle1)*np.cos(self.angle1))
        
        dpX = (force-miC+fF+fF1)/(trolley_weight+mF+mF1)
        dpangle = -0.75*(dpX*np.cos(self.angle)+pom21+pom11)/stick_length
        dpangle1 = -0.75*(dpX*np.cos(self.angle1)+pom22+pom12)/stick_length1
        
        self.pX = self.pX + dpX*time
        self.pangle = self.pangle + dpangle*time
        self.pangle1 = self.pangle1 + dpangle1*time
        
        self.position = self.position + self.pX*time
        self.angle = self.angle + self.pangle*time
        self.angle1 = self.angle1 + self.pangle1*time
        self.time += time


# the number of data entering the network - 6 (I consider so many elements 
# from the above class as important in the process of learning the network, 
# they will be thrown into the X variable)
num_of_x = 6

# fi function - according to mathematical formulas
def fi(x,c,s)->float:
		# I added this condition here to not to waste time - if the number of this 
		# power is too large, we return a small number
		# the reason is that then np.dot() cannot handle matrix multiplication
    if -math.pow(((x-c)/(s)), 2) < -1e7:
        return 1

    return math.exp(-math.pow(((x-c)/(s)), 2)) 

# PNN - predicting how we should go
# the network gets at the entrance:
# X - values ​​that influence the learning process
# c - fi function centers in radial neurons (each neuron has its own center)
#     The centers are randomized as well as the weights to find the best 
#     combination of parameters
# s - sigma parameter in radial neurons (each neuron has its own sigma)
#     The sigmas are randomized as well as the weights to find the best 
#     combination of parameters
# The output of the network really remains to our own interpretation - 
# I interpreted it so that since we get a number in the range <0.1> 
# on the output, we can subtract 0.5 from it and treat it as an added 
# value to the force, e.g.
# we get 0.7 - then 0.5 is subtracted - we get 0.2 - the trolley gets 
# an extra force of 0.2
# we get 0.1 - we subtract 0.5 - we get -0.4 - the force affecting the 
# trolley is reduced by 0.4

# The neural network learns from:
# X - [p.position, p.angle, p.angle1, p.pX, p.pangle, p.pangle1] - 6 values
# c,s,W - values ​​that are recalculating to maximize the reward function
# time - that is, the time it took the trolley with the pendulum without tipping 
# it over - THIS IS OUR REWARD FUNCTION - THE LONGER THE TIME, THE BETTER
def predict(X,c,s,W)->float:

    # creating a matrix for FI - 2x 3 radial neurons
    FI = np.zeros([(int)(num_of_x/3),(int)(num_of_x/2)])
    for i in range((int)(num_of_x/3)):
        for j in range((int)(num_of_x/2)):

            # FI[0,0] = fi(X[0],c[0],s[0])
            # FI[0,1] = fi(X[1],c[1],s[1])
            # FI[0,2] = fi(X[2],c[2],s[2])
            # FI[1,0] = fi(X[3],c[3],s[3])
            # FI[1,1] = fi(X[4],c[4],s[4])
            # FI[1,2] = fi(X[5],c[5],s[5])
            FI[i,j] = fi(X[i*(int)(num_of_x/2)+j],c[i*(int)(num_of_x/2)+j],s[i*(int)(num_of_x/2)+j])

    # two sums Y - the result is the product of the three radial neurons
    # after these calculations there is Y with 2 values
    Y = np.ones(2)
    for i in range(2):
        for j in range((int)(num_of_x/2)):
            Y[i] *= FI[i,j]

    # sum is the product of the matrix Y with the weight matrix W
		# the result is two numbers Sum[1], Sum[2]
    Sum = np.dot(Y,W)

		# this is a response to errors - in some cases (I don't know why) the 
		# network has the second value cleared, and the result of the network is 
		# Sum[0]/Sum[1], which gives us an error and the program did not stop, 
		# but further training did not proceed correct. The solution here is to 
		# return 0 in this case - so according to the network we're slowing down 
		# a bit at this point. I know it's not logical, but if I gave 0.5 
		# (no change) or e.g. random.random() it stopped working xD so we just 
		# brake and everything works
    if Sum[1] == 0:
        return 0

    # prediction result as one number
    return (Sum[0]/Sum[1])

def main():

    # values ​​for simulated annealing
    #
    # initial temperature:   90
    # final temperature:     0.1
    # multiplier:            0.99
    #
    # for 0.99 etc it will take a long time - I recommend 0.9 for testing
    #
    # btw. it happens that the algorithm crashes at some point and does not 
		# start anymore - I do not know what it results from, so it is most 
		# sensible to run it for 0.9 or less - less likely to freeze.
    initial_temp = 90
    current_temp = initial_temp
    final_temp = .1
    alpha = 0.9

    # we create an object of the Pendulum class
    p = Pendulum()

    # 6 center values ​​- 1 for each radial neuron
    # we randomize in the range <-0.5, 0.5>
    c = np.random.random(num_of_x)-1/2

    # 6 sigma values ​​- 1 for each radial neuron
    # we randomize ints in the range <1.9>
    sigma = np.random.randint(low=1, high=10, size=num_of_x)

    # weights - we draw two, two have the value 1
    W = np.array([[np.random.random(), np.random.random()], [1,1]])

    # the above draw ranges are up to us; sigma can be np.random.random(). 
		# What I noticed, the best results come out of this combination

		# here we have the first movement of the vehicle
		# This first move works on completely random weights - after this 
		# action we will know, how much did a trolley driven with these weights.
    force = 0

    # we push the trolley by force
    p.run(force)

    # poz_last i pX_last - values ​​associated with the displacement of the trolley.
    # in one iteration we check how much the trolley has moved
    poz_last = 0
    pX_last = 0

    # tu we check the condition whether the pendulum angle is in the range (-90, 90)
    while (p.get_angle() <= 90 and p.get_angle() >= -90):

        # we check how much the trolley has moved in this iteration
        poz_diff = p.position - poz_last
        pX_diff = p.pX - pX_last

        # we save the current position as last
        poz_last = p.position
        pX_last = p.pX

        # we create an X that will depend on learning
        X = np.array([poz_diff, p.angle, p.angle1, pX_diff, p.pangle, p.pangle1])
            
        # predicting force change and appending it to force variable
        force += predict(X,c,sigma,W) - 0.5

        # push the trolley
        p.run(force)
    
    # we take the time, how much the trolley rode on random values c,s,W
    time = p.get_time()   

    # here we start simulated annealing
    while (current_temp > final_temp):

        # write the current temperature - we know in which iteration the 
				# best solution was found
        print(f'temperature: {format(current_temp, ".6f")}', end='\t')

        # randomize new values of c, sigma, W
        new_c = np.random.random(num_of_x)-0.5
        new_sigma = np.random.randint(low=1, high=10, size=num_of_x)

        # in the weight matrix it should be remembered that we do not 
				# overwrite ones
        new_W = np.copy(W)
        for i in range(len(W)):
            for j in range(len(W[i])):
                if(W[i,j] != 1):
                    new_W[i,j] += np.random.random()-0.5

        # we remove the previous trolley - we already have its result (time), 
				# so we do not need it
        del(p)

        # we create a new trolley - time and all forces acting on it are reset
        p = Pendulum()

        # randomize new force
        force = np.random.randint(-10,10)

        # push the trolley
        p.run(force)

        # lists to make plots
        # x  - time
        # y3 - force acting on the trolley over time
        # y2 - position in time (the trolley moves in 2D)
        # y  - the angle of the pendulum over time
        x_plot = []
        y3_plot = []
        y2_plot = []
        y_plot = []


        # as long as the pendulum has not tipped over, we calculate the result 
				# of our reward function - that is, the amount of time until the 
				# pendulum tipped over
        # this is the result on the basis of which we will choose better and 
				# better solutions with simulated annealing
				
        poz_last = 0
        pX_last = 0

        # until the pendulum tipped over
        while (p.get_angle() <= 90 and p.get_angle() >= -90):

            # we calculate how much the trolley has moved
            poz_diff = p.position - poz_last
            px_diff = p.pX - pX_last

            poz_last = p.position
            pX_last = p.pX

            X = np.array([poz_diff, p.angle, p.angle1, px_diff, p.pangle, p.pangle1])

            # this is where we subtract to 0.5
            #
            # if the predict() method takes longer than 10 seconds, it means 
						# that the result is such large numbers that python cannot cope 
						# with it (especialy math.pow(), exp() and np.dot() methods). 
						# We add random()
            try:
                with time_limit(10):
                    force += predict(X,new_c,new_sigma,new_W) - 0.5
            except TimeoutException as e:
                force += np.random.random() - 0.5

            # force update
            try:
                with time_limit(10):
                    p.run(force)
            except TimeoutException as e:
                p.run(0)

            # append values to plot lists
            x_plot.append(p.get_time())
            y3_plot.append(force)
            y2_plot.append(p.get_position())
            y_plot.append(p.get_angle())

        # the pendulum tipped over - we take the time of the trolley move
        new_time = p.get_time()

        # the probability value needed for the simulated annealing
        check_propability = np.random.random()

        # error handling - if a method overflow error occurs, it means that 
				# the value is too large - we give infinity, thanks to which the 
				# simulated annealing condition will not be met, because <0.1> 
				# is not greater than inf
        try:
            case = math.exp(-(new_time - time) / current_temp)
        except OverflowError:
            case = float('inf')
        
        # condition for accepting a case:
        #   - time is better 
        #   - the probability from the formula from simulated annealing is 
				#     less than our random number     
        if new_time > time or check_propability > case:

            # we enter the results into last_good_configuration, if we 
						# have the best result at the moment
            if new_time > time:
                last_good_configuration['time'] = new_time
                last_good_configuration['sigma'] = new_sigma
                last_good_configuration['centers'] = new_c
                last_good_configuration['weights'] = new_W
                last_good_configuration['x_plot'] = x_plot
                last_good_configuration['y_plot'] = y_plot
                last_good_configuration['y2_plot'] = y2_plot
                last_good_configuration['y3_plot'] = y3_plot

            # new values:
            time = new_time
            W = new_W
            sigma = new_sigma
            c = new_c

            print(f'NEW RESULT! Trolley moving time: {time}')
        
        else:
            print()

        
        # we multiply the temperature by the multiplier
        current_temp *= alpha
    
    # after while() we have in last_good_configuration the values ​​at which 
		# the trolley drove the longest - let's list them and make plots from them
    print(f"\nTrolley moving time:\n{last_good_configuration['time']}\n")
    print(f"Weights:\n{last_good_configuration['weights']}\n")
    print(f"Centers of radial neurons:\n{last_good_configuration['centers']}\n")
    print(f"Sigmas of radial neurons:\n{last_good_configuration['sigma']}\n")
    print("Plots:\n\n")
    
    plt.figure(figsize=(20,10))
    plt.title("The movement of the pendulum over time (bottom to top)")
    plt.plot(last_good_configuration['y_plot'], last_good_configuration['x_plot'])
    plt.show()
    print()
    plt.figure(figsize=(20,10))
    plt.title("The movement of the trolley over time (bottom to top)")
    plt.plot(last_good_configuration['y2_plot'], last_good_configuration['x_plot'])
    plt.show()
    print()     
    plt.figure(figsize=(20,10))
    plt.title("Force acting on the trolley (bottom to top)")
    plt.plot(last_good_configuration['y3_plot'], last_good_configuration['x_plot'])
    plt.show()
    print() 

if __name__ == "__main__":
    main()