import matplotlib.pyplot as plt

class Plot:
    # lists to make plots
    # x  - time
    # y[2] - force acting on the cart over time
    # y[1] - position in time (the cart moves in 2D)
    # y[0] - the angle of the pendulum over time
    def __init__(self):
        self.plot_reset()

    def plot_reset(self):
        self.x = []
        self.y = [[],[],[]]

    @staticmethod
    def set_plot_size(w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)
