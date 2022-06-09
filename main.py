# Important 
# If at some point the program gets stuck, you can pause it in colab, 
# because the best result data will still be saved in Pendulum.last_good_configuration()
# method so you can still plot it with the last lines from the code - just copy 
# the last +-20 lines to the next section, stop the execution of that piece of 
# code and launch the next one
import numpy as np
import matplotlib.pyplot as plt
import math
from plots import Plot
from timer import time_limit
from trolley import Pendulum
from customAlgorithms import SA, PNN

def main():
    # we create an object of the Pendulum class
    p = Pendulum()
    pnn = PNN()
    plot = Plot()
    sa = SA()

    # 6 center values ​​- 1 for each radial neuron
    # we randomize in the range <-0.5, 0.5>
    pnn.centers = pnn.get_random_centers()

    # 6 sigma values ​​- 1 for each radial neuron
    # we randomize ints in the range <1.9>
    pnn.sigmas = pnn.get_random_sigmas()

    # weights - we draw two, two have the value 1
    pnn.weights = pnn.get_random_weights()

    # the above draw ranges are up to us; sigma can be np.random.random(). 
		# What I noticed, the best results come out of this combination

		# here we have the first movement of the vehicle
		# This first move works on completely random weights - after this 
		# action we will know, how much did a trolley driven with these weights.

    # we push the cart by force
    p.run()

    # tu we check the condition whether the pendulum angle is in the range (-90, 90)
    while (p.get_angle() <= 90 and p.get_angle() >= -90):

        # we check how much the cart has moved in this iteration
        poz_diff = p.get_position() - p.last_position
        pX_diff = p.get_pX() - p.last_pX

        # we save the current position as last
        p.last_position = p.get_position()
        p.last_pX = p.get_pX()

        # we create an X that will depend on learning
        X = np.array([poz_diff, p.get_angle(), p.get_angle1(), pX_diff, p.get_pangle(), p.get_pangle1()])
            
        # predicting force change and appending it to force variable
        p.force += pnn.predict(X,pnn.centers,pnn.sigmas,pnn.weights) - 0.5

        # push the trolley
        p.run()  

    Pendulum.best_time = p.time
    
    print(f'Time with random weights: {Pendulum.best_time}', 
          end="\n------------------------------------------\n")

    # here we start simulated annealing
    sa.start(pnn,p,plot)
    
    # after while() we have in Pendulum.last_good_configuration the values ​​at which 
		# the trolley drove the longest - let's list them and make plots from them
    print(f"\nTrolley moving time:\n{Pendulum.last_good_configuration['time']}\n")
    print(f"Weights:\n{Pendulum.last_good_configuration['weights']}\n")
    print(f"Centers of radial neurons:\n{Pendulum.last_good_configuration['centers']}\n")
    print(f"Sigmas of radial neurons:\n{Pendulum.last_good_configuration['sigma']}\n")
    print("Plots:\n\n")
    
    figure, axis = plt.subplots(3, 1)
      
    titles = ["The movement of the pendulum", "The movement of the trolley", "Force acting on the trolley"]
    y_plot_list = ["y_plot", "y2_plot", "y3_plot"]

    for i, title, y in zip([0,1,2], titles, y_plot_list):
        axis[i].plot(Pendulum.last_good_configuration[y], Pendulum.last_good_configuration['x_plot'])
        axis[i].set_title(f"{title} over time (bottom to top)")
        axis.size
    Plot.set_plot_size(15,8)
    plt.show()
    print()

if __name__ == "__main__":
    main()
