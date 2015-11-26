import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import operator

import Tkinter as Tk
import sys
import numpy as np

from genetic_algorithm import GA

class FitnessFunction:

    def getNumberOfVariables(self):
        raise NotImplementedError()

    def getVariableRange(self):
        raise NotImplementedError()

    def getFunctionRange(self):
        raise NotImplementedError()

    def getFunctionData(self):
        raise NotImplementedError()

    def getFitness(self, vars):
        raise NotImplementedError()

class FitnessFunction1(FitnessFunction):

    def __init__(self):
        self.numberOfVariables = 1
        self.variableRange = np.array([[-5.0,1.0]])

        delta = 0.1
        limit = np.fix((self.variableRange[0][1]-self.variableRange[0][0])/delta)+1

        x = np.arange(self.variableRange[0][0],self.variableRange[0][1]+delta,delta)
        y = np.zeros(limit)

        for j in range(0,int(limit),1):
            y[j] = self.getFitness(np.array([x[j]]))

        self.functionData = [x, y]

    def getNumberOfVariables(self):
        return self.numberOfVariables

    def getVariableRange(self):
        return self.variableRange

    def getFunctionRange(self):
        return [[-5,1],[-7,2.5]]

    def getFunctionData(self):
        return self.functionData

    def getFitness(self, vars):
        x = vars[0]
        return -x**2-4*x-2

class FitnessFunction2(FitnessFunction):

    def __init__(self):
        self.numberOfVariables = 1
        self.variableRange = np.array([[-2.0,2.5]])

        delta = 0.1
        limit = np.fix((self.variableRange[0][1]-self.variableRange[0][0])/delta)+1

        x = np.arange(self.variableRange[0][0],self.variableRange[0][1]+delta,delta)
        y = np.zeros(limit)

        for j in range(0,int(limit),1):
            y[j] = self.getFitness(np.array([x[j]]))

        self.functionData = [x, y]

    def getNumberOfVariables(self):
        return self.numberOfVariables

    def getVariableRange(self):
        return self.variableRange

    def getFunctionRange(self):
        return [[-2,2.5],[-9,0]]

    def getFunctionData(self):
        return self.functionData

    def getFitness(self, vars):
        x = vars[0]
        return -x**4+x**3+4*x**2-2*x-5

class FitnessFunction3(FitnessFunction):

    def __init__(self):
        self.numberOfVariables = 2
        self.variableRange = np.array([[-1,1],[-2,0]])

        delta = 0.05
        maxRange = max(self.variableRange[0][1]-self.variableRange[0][0],self.variableRange[1][1]-self.variableRange[1][0])
        limit = np.fix(maxRange/delta)+1

        x = np.arange(self.variableRange[0][0],self.variableRange[0][1]+delta,delta)
        y = np.arange(self.variableRange[1][0],self.variableRange[1][1]+delta,delta)
        x, y = np.meshgrid(x,y)
        z = np.zeros((limit,limit))

        for j in range(0,int(limit),1):
            for k in range(0,int(limit),1):
                z[j,k] = self.getFitness([x[j,k], y[j,k]])

        self.functionData = [x, y, z]

    def getNumberOfVariables(self):
        return self.numberOfVariables

    def getVariableRange(self):
        return self.variableRange

    def getFunctionRange(self):
        return [[-1,1],[-2,0],[0,0.4]]

    def getFunctionData(self):
        return self.functionData

    def getFitness(self, vars):
        x = vars[0]
        y = vars[1]

        a = x + y + 1
        b = 19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2
        c = 2*x - 3*y
        d = 18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2

        f = (1 + a**2 * b) * (30 + c**2 * d)
        return 1.0/f

class FitnessFunction4(FitnessFunction):

    def __init__(self):
        self.numberOfVariables = 2
        self.variableRange = np.array([[-2,2],[-2,2]])

        delta = 0.1
        maxRange = max(self.variableRange[0][1]-self.variableRange[0][0],self.variableRange[1][1]-self.variableRange[1][0])
        limit = np.fix(maxRange/delta)+1

        x = np.arange(self.variableRange[0][0],self.variableRange[0][1]+delta,delta)
        y = np.arange(self.variableRange[1][0],self.variableRange[1][1]+delta,delta)
        x, y = np.meshgrid(x,y)
        z = np.zeros((limit,limit))

        for j in range(0,int(limit),1):
            for k in range(0,int(limit),1):
                z[j,k] = self.getFitness([x[j,k], y[j,k]])

        self.functionData = [x, y, z]

    def getNumberOfVariables(self):
        return self.numberOfVariables

    def getVariableRange(self):
        return self.variableRange

    def getFunctionRange(self):
        return [[-2,2],[-2,2],[0,1]]

    def getFunctionData(self):
        return self.functionData

    def getFitness(self, vars):
        x = vars[0]
        y = vars[1]

        import math
        num = ((math.sin(x**2+y**2))**2)-0.5
        den = (1+(0.001*(x**2+y**2)))**2
        f = 0.5 - num / den
        return f

class Model(object):
    def __init__(self):
        self.observers = []
        self.data = []
        self.function_data = []
        self.selection_type = 0
        self.fitnessFunction = FitnessFunction1()

    def changed(self, event):
        """Notify the observers. """
        for observer in self.observers:
            observer.update(event, self)

    def add_observer(self, observer):
        """Register an observer. """
        self.observers.append(observer)

class Controller(object):
    def __init__(self, model):
        self.model = model
        self.function = Tk.IntVar()
        self.selection_type = Tk.IntVar()
        self.running = False
        self.done = False

    def pause(self):
        self.running = False
        self.model.changed("state_changed")

    def run(self):
        if self.running == True: return
        self.running = True
        self.model.changed("state_changed")

        while self.model.ga.generation < self.model.ga.numberOfGenerations:
            if self.running == False:
                break
            self.model.ga.Step()
            self.model.data = self.model.ga.data
            self.model.changed("data_changed")

        if self.model.ga.generation >= self.model.ga.numberOfGenerations:
            self.running = False
            self.done = True
            self.model.changed("state_changed")

    def step(self):
        if self.running == False and self.model.ga.generation < self.model.ga.numberOfGenerations:
            self.model.ga.Step()
            self.model.data = self.model.ga.data
            self.model.changed("data_changed")

        if self.model.ga.generation >= self.model.ga.numberOfGenerations:
            self.done = True
            self.model.changed("state_changed")

    def config_changed(self):
        self.running = False
        self.done = False
        self.model.changed("state_changed")
        function = self.function.get()
        populationSize = int(self.populationSize.get())
        numberOfGenes = int(self.numberOfGenes.get())
        crossoverProbability = float(self.crossoverProbability.get())
        mutationProbability = float(self.mutationProbability.get())
        tournamentSelectionProbability = float(self.tournamentSelectionProbability.get())
        tournamentSize = int(self.tournamentSize.get())
        numberOfGenerations = int(self.numberOfGenerations.get())
        numberOfElites = int(self.numberOfElites.get())
        selectionType = self.selection_type.get()
        useElitism = self.use_elitism.get()

        if function == 1:
            self.model.fitnessFunction = FitnessFunction1()
        elif function == 2:
            self.model.fitnessFunction = FitnessFunction2()
        elif function == 3:
            self.model.fitnessFunction = FitnessFunction3()
        else:
            self.model.fitnessFunction = FitnessFunction4()

        numberOfVariables = self.model.fitnessFunction.getNumberOfVariables()
        variableRange = self.model.fitnessFunction.getVariableRange()
        fitnessFunction = self.model.fitnessFunction.getFitness

        self.model.ga = GA(populationSize, numberOfGenes, crossoverProbability, mutationProbability,
            selectionType, tournamentSelectionProbability, tournamentSize, numberOfVariables,
            variableRange, numberOfGenerations, useElitism, numberOfElites, fitnessFunction)

        self.model.data = self.model.ga.data
        self.model.function_data = self.model.fitnessFunction.getFunctionData()
        self.model.changed("config_changed")

class View(object):
    """Test docstring. """
    def __init__(self, root, controller):
        self.controllbar = ControllBar(root, controller)

        f = Figure()
        ax = f.add_subplot(111)
        xyzlim = controller.model.fitnessFunction.getFunctionRange()
        numVars = controller.model.fitnessFunction.getNumberOfVariables()
        if numVars == 1:
            ax.set_xlim((xyzlim[0][0], xyzlim[0][1]))
            ax.set_ylim((xyzlim[1][0], xyzlim[1][1]))
        if numVars == 2:
            ax.set_xlim3d(xyzlim[0])
            ax.set_ylim3d(xyzlim[1])
            ax.set_zlim3d(xyzlim[2])
            ax.set_zlabel('z')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas._tkcanvas.pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
        canvas.get_tk_widget().pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()

        f2 = Figure()
        ax2 = f2.add_subplot(111)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        canvas2 = FigureCanvasTkAgg(f2, master=root)
        canvas2.show()
        canvas2._tkcanvas.pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
        canvas2.get_tk_widget().pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
        toolbar2 = NavigationToolbar2TkAgg(canvas2, root)
        toolbar2.update()

        self.f = f
        self.f2 = f2
        self.ax = ax
        self.ax2 = ax2
        self.canvas = canvas
        self.canvas2 = canvas2
        self.controller = controller
        self.root = root

    def update(self, event, model):

        if event == "state_changed":

            if self.controller.done:
                self.controllbar.stepButton.config(state=Tk.DISABLED)
                self.controllbar.runButton.config(state=Tk.DISABLED)
                self.controllbar.pauseButton.config(state=Tk.DISABLED)
            else:
                if self.controller.running:
                    self.controllbar.stepButton.config(state=Tk.DISABLED)
                    self.controllbar.runButton.config(state=Tk.DISABLED)
                    self.controllbar.pauseButton.config(state=Tk.NORMAL)
                else:
                    self.controllbar.stepButton.config(state=Tk.NORMAL)
                    self.controllbar.runButton.config(state=Tk.NORMAL)
                    self.controllbar.pauseButton.config(state=Tk.DISABLED)

        if event == "config_changed":
            xyzlim = model.fitnessFunction.getFunctionRange()
            numVars = model.fitnessFunction.getNumberOfVariables()
            if numVars == 1:
                self.f.delaxes(self.ax)
                ax = self.f.add_subplot(111)
                ax.set_xlim((xyzlim[0][0], xyzlim[0][1]))
                ax.set_ylim((xyzlim[1][0], xyzlim[1][1]))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                self.ax = ax
                self.surf = self.ax.plot(model.function_data[0],model.function_data[1])
                self.ax.relim()
                self.ax.autoscale_view()
                self.data = model.data
                self.lines = [self.ax.plot(dat[0], dat[1], 'o')[0] for dat in self.data]
            elif numVars == 2:
                self.f.delaxes(self.ax)
                ax = self.f.add_subplot(111,projection='3d')
                ax.set_xlim3d(xyzlim[0])
                ax.set_ylim3d(xyzlim[1])
                ax.set_zlim3d(xyzlim[2])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                self.ax = ax
                self.surf = self.ax.plot_wireframe(model.function_data[0],model.function_data[1],model.function_data[2], rstride=1, cstride=1, color='b')
                #self.surf = self.ax.plot_surface(model.function_data[0],model.function_data[1],model.function_data[2], rstride=1, cstride=1, cmap=cm.spectral, linewidth=0, antialiased=False )
                self.ax.relim()
                self.ax.autoscale_view()
                self.data = model.data
                self.lines = [self.ax.plot(dat[[0]], dat[[1]], dat[[2]], 'o')[0] for dat in self.data]

            self.ax2.clear()
            self.ax2.set_xlabel('Generation')
            self.ax2.set_ylabel('Fitness')
            self.lines2, = self.ax2.plot([],[],'g-')
            self.lines3, = self.ax2.plot([],[],'r-')
            self.lines2.set_xdata(np.append(self.lines2.get_xdata(), model.ga.generation))
            self.lines2.set_ydata(np.append(self.lines2.get_ydata(), model.ga.maximumFitness))
            self.lines3.set_xdata(np.append(self.lines3.get_xdata(), model.ga.generation))
            self.lines3.set_ydata(np.append(self.lines3.get_ydata(), model.ga.averageFitness))
            #self.ax2.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
            best = model.data[model.ga.bestIndividualIndex]
            if model.ga.numberOfVariables == 1:
                self.ax2.legend(('Max Fitness f({0:.3f})={1:.3f}'.format(best[0], best[1]), 'Avg Fitness ({0:.3f})'.format(model.ga.averageFitness)), loc='lower left')
            else:
                self.ax2.legend(('Max Fitness f({0:.3f},{1:.3f})={2:.3f}'.format(best[0], best[1], best[2]), 'Avg Fitness ({0:.3f})'.format(model.ga.averageFitness)), loc='lower left')
            self.ax2.relim()
            self.ax2.autoscale_view()

        if event == "data_changed":
            numVars = model.fitnessFunction.getNumberOfVariables()
            if numVars == 1:
                for p, d in zip(self.lines, model.data):
                    p.set_data(d[0:2])
            elif numVars == 2:
                for p, d in zip(self.lines, model.data):
                    p.set_data(d[0:2])
                    p.set_3d_properties(d[2])

            self.lines2.set_xdata(np.append(self.lines2.get_xdata(), model.ga.generation))
            self.lines2.set_ydata(np.append(self.lines2.get_ydata(), model.ga.maximumFitness))
            self.lines3.set_xdata(np.append(self.lines3.get_xdata(), model.ga.generation))
            self.lines3.set_ydata(np.append(self.lines3.get_ydata(), model.ga.averageFitness))
            best = model.data[model.ga.bestIndividualIndex]
            if model.ga.numberOfVariables == 1:
                self.ax2.legend(('Max Fitness f({0:.3f})={1:.3f}'.format(best[0], best[1]), 'Avg Fitness ({0:.3f})'.format(model.ga.averageFitness)), loc='lower left')
            else:
                self.ax2.legend(('Max Fitness f({0:.3f},{1:.3f})={2:.3f}'.format(best[0], best[1], best[2]), 'Avg Fitness ({0:.3f})'.format(model.ga.averageFitness)), loc='lower left')
            self.ax2.relim()
            self.ax2.autoscale_view()

        self.canvas.draw()
        self.canvas2.draw()
        self.canvas2.flush_events()

    def quit(self):
        self.controller.running = False
        self.root.destroy()
        self.root.quit()

class ControllBar(object):
    def __init__(self, root, controller):
        fm = Tk.Frame(root)

        config_group = Tk.LabelFrame(fm, height=2, bd=1, text="Configuration", relief=Tk.SOLID)

        # Parameter Settings Group
        params_group = Tk.LabelFrame(config_group, height=2, bd=1, text="Basic Parameters", relief=Tk.SOLID)
        # Number Of Generations
        controller.numberOfGenerations = Tk.StringVar()
        controller.numberOfGenerations.set("100")
        nGenerations = Tk.Frame(params_group)
        Tk.Label(nGenerations, text="#Generations:", anchor="e", width=10).pack(side=Tk.LEFT)
        Tk.Entry(nGenerations, width=6, textvariable=controller.numberOfGenerations).pack(side=Tk.LEFT)
        nGenerations.pack()
        # Population Size
        controller.populationSize = Tk.StringVar()
        controller.populationSize.set("10")
        pSize = Tk.Frame(params_group)
        Tk.Label(pSize, text="#Individuals:", anchor="e", width=10).pack(side=Tk.LEFT)
        Tk.Entry(pSize, width=6, textvariable=controller.populationSize).pack(side=Tk.LEFT)
        pSize.pack()
        # Number Of Genes
        controller.numberOfGenes = Tk.StringVar()
        controller.numberOfGenes.set("50")
        nGenes = Tk.Frame(params_group)
        Tk.Label(nGenes, text="#Genes:", anchor="e", width=10).pack(side=Tk.LEFT)
        Tk.Entry(nGenes, width=6, textvariable=controller.numberOfGenes).pack(side=Tk.LEFT)
        nGenes.pack()
        # Crossover Probability
        controller.crossoverProbability = Tk.StringVar()
        controller.crossoverProbability.set("0.8")
        cProb = Tk.Frame(params_group)
        Tk.Label(cProb, text="Pcross:", anchor="e", width=10).pack(side=Tk.LEFT)
        Tk.Entry(cProb, width=6, textvariable=controller.crossoverProbability).pack(side=Tk.LEFT)
        cProb.pack()
        # Mutation Probability
        controller.mutationProbability = Tk.StringVar()
        controller.mutationProbability.set("0.1")
        mProb = Tk.Frame(params_group)
        Tk.Label(mProb, text="Pmut:", anchor="e", width=10).pack(side=Tk.LEFT)
        Tk.Entry(mProb, width=6, textvariable=controller.mutationProbability).pack(side=Tk.LEFT)
        mProb.pack()
        params_group.pack(side=Tk.LEFT, fill=Tk.X, padx=5, pady=5)

        # Selection Group
        selection_group = Tk.LabelFrame(config_group, height=2, bd=1, text="Selection Method", relief=Tk.SOLID)
        Tk.Radiobutton(selection_group, text="Tournament",
                       variable=controller.selection_type, value=0).pack(anchor=Tk.W)
        # Tournament Selection Probability
        controller.tournamentSelectionProbability = Tk.StringVar()
        controller.tournamentSelectionProbability.set("0.75")
        pTour = Tk.Frame(selection_group)
        Tk.Label(pTour, text="Ptour:", anchor="e", width=9).pack(side=Tk.LEFT)
        Tk.Entry(pTour, width=6, textvariable=controller.tournamentSelectionProbability).pack(side=Tk.LEFT)
        pTour.pack()
        # Tournament Size
        controller.tournamentSize = Tk.StringVar()
        controller.tournamentSize.set("4")
        tSize = Tk.Frame(selection_group)
        Tk.Label(tSize, text="Size:", anchor="e", width=9).pack(side=Tk.LEFT)
        Tk.Entry(tSize, width=6, textvariable=controller.tournamentSize).pack(side=Tk.LEFT)
        tSize.pack()
        Tk.Radiobutton(selection_group, text="Roulette Wheel",
                       variable=controller.selection_type, value=1).pack(anchor=Tk.W)
        selection_group.pack(side=Tk.LEFT, fill=Tk.X, padx=5, pady=5)
        # Elite Group
        elite_group = Tk.LabelFrame(config_group, height=2, bd=1, text="Elitism", relief=Tk.SOLID)
        controller.use_elitism = Tk.BooleanVar()
        controller.use_elitism.set(True)
        Tk.Checkbutton(elite_group, text="Enabled", variable=controller.use_elitism).pack(anchor=Tk.W)
        # Number Of Elites
        controller.numberOfElites = Tk.StringVar()
        controller.numberOfElites.set("1")
        nElite = Tk.Frame(elite_group)
        Tk.Label(nElite, text="#Elite:", anchor="e", width=9).pack(side=Tk.LEFT)
        Tk.Entry(nElite, width=6, textvariable=controller.numberOfElites).pack(side=Tk.LEFT)
        nElite.pack()
        elite_group.pack(side=Tk.LEFT, fill=Tk.X, padx=5, pady=5)
        # Function Group
        function_group = Tk.LabelFrame(config_group, height=2, bd=1, text="Test Function", relief=Tk.SOLID)
        rbF1 = Tk.Radiobutton(function_group, text="f1", variable=controller.function, value=1)
        rbF1.pack(side=Tk.LEFT)
        Tk.Radiobutton(function_group, text="f2", variable=controller.function, value=2).pack(side=Tk.LEFT)
        Tk.Radiobutton(function_group, text="f3", variable=controller.function, value=3).pack(side=Tk.LEFT)
        Tk.Radiobutton(function_group, text="f4", variable=controller.function, value=4).pack(side=Tk.LEFT)
        rbF1.select()
        function_group.pack(side=Tk.LEFT, fill=Tk.X, padx=5, pady=5)
        # Set Button
        Tk.Button(config_group, text='Set', width=5, command=controller.config_changed).pack(side=Tk.BOTTOM, padx=5, pady=5)

        # Button Group
        right_frame = Tk.Frame(fm)
        button_group = Tk.LabelFrame(right_frame, height=2, bd=1, text="Control", relief=Tk.SOLID)
        self.stepButton = Tk.Button(button_group, text='Step', width=5, command=controller.step)
        self.stepButton.pack(side=Tk.LEFT)
        self.runButton = Tk.Button(button_group, text='Run', width=5, command=controller.run)
        self.runButton.pack(side=Tk.LEFT)
        self.pauseButton = Tk.Button(button_group, text='Pause', width=5, command=controller.pause)
        self.pauseButton.pack(side=Tk.LEFT)
        self.resetButton = Tk.Button(button_group, text='Reset', width=5, command=controller.config_changed)
        self.resetButton.pack(side=Tk.LEFT)
        button_group.pack(side=Tk.BOTTOM, fill=Tk.X, padx=5, pady=5)

        right_frame.pack(side=Tk.BOTTOM)
        config_group.pack(side=Tk.RIGHT)
        fm.pack(side=Tk.LEFT, fill=Tk.X, padx=60, pady=5)

def main(argv):
    root = Tk.Tk()
    model = Model()
    controller = Controller(model)
    root.wm_title("Genetic Algorithm GUI")
    view = View(root, controller)
    root.protocol('WM_DELETE_WINDOW', view.quit)
    model.add_observer(view)
    controller.config_changed()
    Tk.mainloop()

if __name__ == "__main__":
    main(sys.argv)
