import numpy as np
import math
from trolley import Pendulum
from timer import time_limit

class PNN:
    centers = None
    weights = None
    sigmas = None
    new_centers = None
    new_weights = None
    new_sigmas = None

    def __init__(self):
        # the number of data entering the network - 6 (I consider so many elements 
        # from the above class as important in the process of learning the network, 
        # they will be thrown into the X variable)
        self.num_of_x = 6

    # fi function - according to mathematical formula
    def fi(self, x,c,s)->float:
        # I added this condition here to not to waste time - if the number of this 
        # power is too large, we return a small number
        # the reason is that then np.dot() cannot handle matrix multiplication
        if -math.pow(((x-c)/(s)), 2) < -1e7:
            return 1

        return math.exp(-math.pow(((x-c)/(s)), 2)) 
    
    def get_random_centers(self):
        return np.random.random(self.num_of_x)-1/2

    def get_random_sigmas(self):
        return np.random.randint(low=1, high=10, size=self.num_of_x)
      
    def get_random_weights(self):
        return np.array([[np.random.random(), np.random.random()], [1,1]])

    def update_weights(self):
        new_weights = np.copy(self.weights)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if(self.weights[i,j] != 1):
                    new_weights[i,j] += np.random.random()-0.5

        return new_weights

    # neuro-fuzzy network - predicting how we should go
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
    # X - [p.get_position(), p.get_angle(), p.get_angle1(), p.get_pX(), p.get_pangle(), p.get_pangle1()] - 6 values
    # c,s,W - values ​​that are recalculating to maximize the reward function
    # time - that is, the time it took the cart with the pendulum without tipping 
    # it over - THIS IS OUR REWARD FUNCTION - THE LONGER THE TIME, THE BETTER
    def predict(self, X,c,s,W)->float:

        # creating a matrix for FI - 2x 3 radial neurons
        FI = np.zeros([(int)(self.num_of_x/3),(int)(self.num_of_x/2)])
        for i in range((int)(self.num_of_x/3)):
            for j in range((int)(self.num_of_x/2)):

                # FI[0,0] = fi(X[0],c[0],s[0])
                # FI[0,1] = fi(X[1],c[1],s[1])
                # FI[0,2] = fi(X[2],c[2],s[2])
                # FI[1,0] = fi(X[3],c[3],s[3])
                # FI[1,1] = fi(X[4],c[4],s[4])
                # FI[1,2] = fi(X[5],c[5],s[5])
                FI[i,j] = self.fi(X[i*(int)(self.num_of_x/2)+j],c[i*(int)(self.num_of_x/2)+j],s[i*(int)(self.num_of_x/2)+j])

        # two sums Y - the result is the product of the three radial neurons
        # after these calculations there is Y with 2 values
        Y = np.ones(2)
        for i in range(2):
            for j in range((int)(self.num_of_x/2)):
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

class SA:
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
    alpha = 0.8

    def start(self, pnn, p, plot):
        while (self.current_temp > self.final_temp):

            # write the current temperature - we know in which iteration the 
            # best solution was found
            print(f'temperature: {format(self.current_temp, ".6f")}', end='\t')

            # randomize new values of c, sigma, W
            pnn.new_centers = pnn.get_random_centers()
            pnn.new_sigmas = pnn.get_random_sigmas()
            pnn.new_weights = pnn.update_weights()

            # we remove the previous cart - we already have its result (time), 
            # so we do not need it
            del(p)

            # we create a new trolley - time and all forces acting on it are reset
            p = Pendulum()

            # randomize new force
            p.force = np.random.randint(-10,10)

            # push the trolley
            p.run()

            # as long as the pendulum has not tipped over, we calculate the result 
            # of our reward function - that is, the amount of time until the 
            # pendulum tipped over
            # this is the result on the basis of which we will choose better and 
            # better solutions with simulated annealing
            
            plot.plot_reset();

            p.last_position = 0
            p.last_pX = 0

            # until the pendulum tipped over
            while (p.get_angle() <= 90 and p.get_angle() >= -90):

                # we calculate how much the cart has moved
                poz_diff = p.get_position() - p.last_position
                px_diff = p.get_pX() - p.last_pX

                p.last_position = p.get_position()
                p.last_pX = p.get_pX()

                X = np.array([poz_diff, p.get_angle(), p.get_angle1(), px_diff, p.get_pangle(), p.get_pangle1()])

                # this is where we subtract to 0.5
                #
                # if the predict() method takes longer than 10 seconds, it means 
                # that the result is such large numbers that python cannot cope 
                # with it (especialy math.pow(), exp() and np.dot() methods). 
                # We add random()
                try:
                    with time_limit(10):
                        p.force += pnn.predict(X,pnn.new_centers,pnn.new_sigmas,pnn.new_weights) - 0.5
                except TimeoutException as e:
                    p.force += np.random.random() - 0.5

                # force update
                try:
                    with time_limit(10):
                        p.run()
                except TimeoutException as e:
                    p.run(0)

                # append values to plot lists
                plot.x.append(p.time)
                plot.y[0].append(p.force)
                plot.y[1].append(p.get_position())
                plot.y[2].append(p.get_angle())

            # the pendulum tipped over - we take the time of the trolley move from p.time

            # error handling - if a method overflow error occurs, it means that 
            # the value is too large - we give infinity, thanks to which the 
            # simulated annealing condition will not be met, because <0.1> 
            # is not greater than inf
            try:
                case = math.exp(-(Pendulum.best_time - p.time) / self.current_temp)
            except OverflowError:
                case = float('inf')
            # condition for accepting a case:
            #   - time is better 
            #   - the probability from the formula from simulated annealing is 
            #     less than our random number     

            if np.random.random() < case:

                # we enter the results into Pendulum.last_good_configuration, if we 
                # have the best result at the moment
                if Pendulum.best_time < p.time:
                    Pendulum.last_good_configuration['time'] = p.time
                    Pendulum.last_good_configuration['sigma'] = pnn.new_sigmas
                    Pendulum.last_good_configuration['centers'] = pnn.new_centers
                    Pendulum.last_good_configuration['weights'] = pnn.new_weights
                    Pendulum.last_good_configuration['x_plot'] = plot.x
                    Pendulum.last_good_configuration['y3_plot'] = plot.y[0]
                    Pendulum.last_good_configuration['y2_plot'] = plot.y[1]
                    Pendulum.last_good_configuration['y_plot'] = plot.y[2]

                # new values:
                Pendulum.best_time = p.time
                pnn.weights = pnn.new_weights
                pnn.sigmas = pnn.new_sigmas
                pnn.centers = pnn.new_centers

                print(f'NEW RESULT! Trolley moving time: {format(Pendulum.best_time, ".6f")}')
            
            else:
                print()

            
            # we multiply the temperature by the multiplier
            self.current_temp *= self.alpha

