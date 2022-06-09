# Probabilistic neural network for trolley with pendulum on itself

## Description

The program creates the PNN and teaches the network to drive the trolley with pendulum as long as possible without the pendulum toppling over. Simulated annealing was chosen for the optimization of the weight matrix.

##Simulation Annealing

Pseudocode:
```
Let s = s0
For k = 0 through kmax (exclusive):
	T ← temperature( 1 - (k+1)/kmax )
	Pick a random neighbour, snew ← neighbour(s)
	If P(E(s), E(snew), T) ≥ random(0, 1):
		s ← snew
Output: the final state s
```

## Animation showing how the algorithm works

![Simulation Annealing](https://upload.wikimedia.org/wikipedia/commons/d/d5/Hill_Climbing_with_Simulated_Annealing.gif)


## Probabilistic Neural Network

I used PNN from screen (with four const ones)

![PNN](https://gcdnb.pbrd.co/images/WC49nmW1VC67.png?o=1)


## Getting Started

### Executing program

Just run the program. 
```
python main.py
```
It also works with Jupyter Notebook e.g.
```
colab.research.google.com
```

Remember about libraries
```
pip install mathplotlib
pip install numpy
```

## Results:
![CLI](https://i.ibb.co/mHGg8G2/result-cli.jpg)

![plots](https://i.ibb.co/51rT0Px/result-plots.jpg)


## Author

[@MatekStatek](https://twitter.com/matekstatek)

## Version History

* 1.0
    * All done
