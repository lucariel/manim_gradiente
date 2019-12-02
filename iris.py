
from manimlib.imports import *
import os
import pandas as pd
import pyclbr
from math import cos, sin, pi
from random import randrange
from sklearn.datasets import load_iris 


class Graphing(GraphScene):
    CONFIG = {
        "x_min": -1,
        "x_max": 7,
        "y_min": -1,
        "y_max": 4,
        "x_axis_width": 7,
        "y_axis_height": 3.5,
        "graph_origin": ORIGIN,
        "function_color": WHITE,
        "axes_color": BLUE,
        "graph_origin": 5 * DOWN + 0.5 * LEFT,
    }

    iris=load_iris()
    iris = pd.DataFrame(iris.data)
    iris.columns = ['petal_length','petal_width', 'sepal_length', 'sepal_width']

    a = 0
    b = 0
    costo = 0

    def construct(self):
        #Make graph
        self.graph_origin = -0.5 * DOWN + 0.5 * LEFT
        self.setup_axes(animate=True)
        func_graph=self.get_graph(self.func_to_graph,self.function_color)
        graph_lab = self.get_graph_label(func_graph, label = "{}x+{}=y".format(self.a , self.b))
        

        x = self.coords_to_point(1, self.func_to_graph(1))
        y = self.coords_to_point(0, self.func_to_graph(1))

        points = []

        theta = np.array([self.a, self.b])
        self.iris['x0'] = 1
        X = self.iris.iloc[:,[4,2]]
        
        y = np.array(self.iris.iloc[:,3])
        num_iter = 15
        alpha = 0.01
        m = len(X)

        xs = X.sepal_length.tolist()
        ys = y
        
        for i in range(len(ys)):
            points.append(Dot(self.coords_to_point((xs[i]),(ys[i])), color = GREEN).scale(0.5))

        
        dots = VGroup(*points)
        self.play(FadeIn(dots))
        

        func_graph1=self.get_graph(self.func_to_graph,self.function_color)
        

        itertext1 = TextMobject("iter: 0").scale(0.7)
        itertext1.next_to(func_graph, 8*UP)
        self.play(ShowCreation(func_graph1),ShowCreation(itertext1))

        first_line = TextMobject("El objetivo del").scale(0.70)
        second_line = TextMobject("algoritmo es actualizar en $y = \\theta_{0} + \\theta_{1}*x$").scale(0.7) 
        third_line = TextMobject("los params $\\theta_{0}$ y $\\theta_{1}$ con las formulas:").scale(0.7)
        first_eq_theta0 = TextMobject("$\\theta_0 := \\theta_0 - \\frac{\\partial J}{\\partial \\theta_0}$"+"$= {}$".format(np.round(self.b,3))).scale(0.7)
        first_eq_theta1 = TextMobject("$\\theta_1 := \\theta_1 - \\frac{\\partial J}{\\partial \\theta_1}$"+"$= {}$".format(np.round(self.a,3))).scale(0.7)
        
        fourth_line = TextMobject("Para minimizar el costo $J$:").scale(0.7)
        second_eq = TextMobject("$J(\\theta_0, \\theta_1) = {}$".format(np.round(self.costo,3))).scale(0.7)


        first_line.next_to(func_graph1, 6*LEFT)
        first_line.shift(2*UP)
        second_line.next_to(first_line, DOWN)
        third_line.next_to(second_line, DOWN)
        first_eq_theta0.next_to(third_line, DOWN*3)
        first_eq_theta1.next_to(first_eq_theta0, DOWN)
        fourth_line.next_to(first_eq_theta1, 4*DOWN)
        second_eq.next_to(fourth_line, 3*DOWN)
        graph_lab.scale(0.5)
        graph_lab.next_to(func_graph, DOWN)

        #self.play(ShowCreation(func_graph1))

        self.graph_origin = 3.7 * DOWN + 0.5 * LEFT
        self.x_axis_label = 'iter'
        self.y_axis_label = '$J$'
        self.x_max = 15
        self.x_axis_width = 7
        self.x_tick_frequency = 1
        self.y_max = 1.5
        self.y_tick_frequency = 0.5
        self.y_axis_height = 4
        self.setup_axes(animate=True)
                

        self.play(Write(first_line),Write(second_line),Write(third_line),Write(first_eq_theta0),Write(first_eq_theta1),Write(second_eq), Write(fourth_line))
        self.costo = self.costFunction(X, y, theta)

        for i in range(0, num_iter):
            itertext = TextMobject("iter: {}".format(i)).scale(0.7)
            itertext.next_to(func_graph, 8*UP)
            self.costo = self.costFunction(X, y, theta)
            temp0 = theta[0]-(alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,0])
            temp1 = theta[1]-(alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,1])
            
            if(i%1==0):
                self.graph_origin = -0.5 * DOWN + 0.5 * LEFT
                self.x_axis_label = '$x$'
                self.y_axis_label = '$y$'
                self.x_max = 7
                self.y_max = 4
                self.y_tick_frequency = 1
                self.y_axis_height = 3.5
                self.x_tick_frequency = 1
                self.x_axis_width = 7
                self.setup_axes(animate=True)
                graph_lab2 = self.get_graph_label(func_graph, label = "({})x+({})=y".format(np.round(self.a,3) , np.round(self.b,3)))
                graph_lab2.scale(0.5)
                graph_lab2.next_to(func_graph, DOWN)
                func_graph2 = self.get_graph(self.func_to_graph,self.function_color)
                second_eq2 = TextMobject("$J(\\theta_0, \\theta_1) = {}$".format(np.round(self.costo,3))).scale(0.7)
                second_eq2.next_to(fourth_line, 3*DOWN)
                first_eq_theta01 = TextMobject("$\\theta_0 := \\theta_0 - \\frac{\\partial J}{\\partial \\theta_0}$"+"$= {} - $".format(np.round(theta[0],3))+"$({})$".format(np.round((alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,0]),3))+"$={}$".format(np.round(temp0,3))).scale(0.6)
                first_eq_theta01.next_to(third_line, DOWN*3)
                first_eq_theta11 = TextMobject("$\\theta_1 := \\theta_1 - \\frac{\\partial J}{\\partial \\theta_1}$"+"$= {} - $".format(np.round(theta[1],3))+"$({})$".format(np.round((alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,1]),3))+"$={}$".format(np.round(temp1,3))).scale(0.6)
                first_eq_theta11.next_to(first_eq_theta0, DOWN)
                self.play(Transform(func_graph1, func_graph2), Transform(graph_lab, graph_lab2),
                    Transform(itertext1, itertext), Transform(second_eq,second_eq2), Transform(first_eq_theta0, first_eq_theta01),
                    Transform(first_eq_theta1, first_eq_theta11))






                self.graph_origin = 3.7 * DOWN + 0.5 * LEFT
                self.x_axis_label = 'iter'
                self.y_axis_label = '$J$'
                self.x_max = 15
                self.x_axis_width = 7
                self.x_tick_frequency = 1
                self.y_max = 1.5
                self.y_tick_frequency = 0.5
                self.y_axis_height = 4
                
                self.setup_axes(animate=True)
                


                J = Dot(self.coords_to_point(i, self.costo), color = RED).scale(0.75)

                self.play(ShowCreation(J))

                

             
            

            theta = np.array([temp0,temp1])
            self.a = theta[1]
            self.b = theta[0]


            #func_graph1 = func_graph2


        



        self.wait(4)
        
            





    def func_to_graph(self, x):
        return (self.a*x+self.b)

    def func_to_graph1(self, x):
        return 0
        


    def gradientDescent(X, y, theta, alpha, num_iter):
        m= len(X)
        for i in range(0, num_iter):
            temp0 = theta[0]-(alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,0])
            temp1 = theta[1]-(alpha/m)*sum((np.dot(X,theta)-y)*X.iloc[:,1])
            theta = np.array([temp0,temp1])
            costFunction(X, y, theta)
        
        return theta

    def costFunction(self, X, y, theta):
        m = len(X)
        predictions = np.dot(X,theta)
        sqrErrors = np.array((predictions-y))**2
        J = 1/(2*m)*np.sum(sqrErrors)
        return J;