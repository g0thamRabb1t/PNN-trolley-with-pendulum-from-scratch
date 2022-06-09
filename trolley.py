import numpy as np

# Here's the code with the calculation for the trolley. The comments below 
# describe what is important in the learning process.
class Pendulum:
    # static
    best_time = None
    # here we will hold the best result and then make a graph of it
    last_good_configuration = dict.fromkeys(['time', 'sigma', 'centers', 'weights', 'x_plot', 'y_plot', 'y2_plot', 'y3_plot'])

    def __init__ (self):
        self.__position = 0
        self.__angle = 4
        self.__angle1 = 0
	
        self.__pX = 0
        self.__pangle = 0
        self.__pangle1 = 0
        self.time = 0
        self.force = 0

        # p.last_position i p.last_pX - values ​​associated with the displacement of the trolley.
        # in one iteration we check how much the cart has moved
        self.last_position = 0
        self.last_pX = 0

        self.stick_length = 0.5
        self.stick_length1 = 0.05
        self.stick_weight = 0.1
        self.stick_weight1 = 0.1
        self.trolley_weight = 1
        self.gravitation = 9.8	
        self.miC = 0.0005
        self.miP = 0.000002
        self.time_gain = 0.02

    def get_position(self):
        return self.__position

    def get_pX(self):
        return self.__pX

    def get_pangle(self):
        return self.__pangle

    def get_pangle1(self):
        return self.__pangle1
    
    def get_angle(self):
        return self.__angle

    def get_angle1(self):
        return self.__angle1

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
    def run(self):
        md = self.stick_weight*self.stick_length
        md1 = self.stick_weight1*self.stick_length1
        
        pom11 = (self.miP * self.__pangle) / md
        pom21 = self.gravitation*np.sin(self.__angle)
        
        pom12 = (self.miP*self.__pangle1)/md1
        pom22 = self.gravitation*np.sin(self.__angle1)
        
        fF = md*self.__angle*self.__angle*np.sin(self.__angle)+0.75*self.stick_weight*np.cos(self.__angle)*(pom11+pom21)
        mF = self.stick_weight*(1 - 0.75*np.cos(self.__angle)*np.cos(self.__angle))
        
        fF1 = md1*self.__angle1*self.__angle1*np.sin(self.__angle1)+0.75*self.stick_weight1*np.cos(self.__angle1)*(pom12+pom22)
        mF1 = self.stick_weight1*(1 - 0.75*np.cos(self.__angle1)*np.cos(self.__angle1))
        
        dpX = (self.force-self.miC+fF+fF1)/(self.trolley_weight+mF+mF1)
        dpangle = -0.75*(dpX*np.cos(self.__angle)+pom21+pom11)/self.stick_length
        dpangle1 = -0.75*(dpX*np.cos(self.__angle1)+pom22+pom12)/self.stick_length1
        
        self.__pX = self.__pX + dpX*self.time_gain
        self.__pangle = self.__pangle + dpangle*self.time_gain
        self.__pangle1 = self.__pangle1 + dpangle1*self.time_gain
        
        self.__position = self.__position + self.__pX*self.time_gain
        self.__angle = self.__angle + self.__pangle*self.time_gain
        self.__angle1 = self.__angle1 + self.__pangle1*self.time_gain
        self.time += self.time_gain
