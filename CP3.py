


import numpy as np
import math
import array
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

def main():
    #Formula
    def ZtoDFormula(H0,Omega_m,Omega_lambda,Omega_k,x):
        return (3*10**5)/H0*(Omega_m*(1+x)**3+Omega_k*(1+x)**2+Omega_lambda)**(-0.5)   


    #Class and Methods
    class Cosmology:
        def __init__(self,H0,Omega_m,Omega_lambda):
            self.H0 = H0
            self.Omega_m = Omega_m
            self.Omega_lambda = Omega_lambda
            self.Omega_k = 1-Omega_m-Omega_lambda
            self.Omega_m_h_square = Omega_m*(H0/100)**2
            self.ifFlat = "true" if self.Omega_k==0 else "false"
            self.modelDescribe = f"Cosmology with H0={H0}, Omega_m={Omega_m}, Omega_lambda={Omega_lambda}, Omega_k={self.Omega_k}"

        def finalFormula(self,x):
            return ZtoDFormula(self.H0,self.Omega_m,self.Omega_lambda,self.Omega_k,x)
    
        def calculateD(self,redshift):
            result,error = quad(self.finalFormula,0,redshift)
            return result
        


        #-----Unit3 New Content-----
        # 4.1---New Methods
        def rectangle(self,redshift,n):
            dx = redshift/(n-1)
            distance = 0
            for i in range(n-1):
                distance += self.finalFormula(i*dx)*dx
            return distance
        
        def trapezoid(self,redshift,n):
            dx = redshift/(n-1)
            distance = (self.finalFormula(0)+self.finalFormula(redshift))/2*dx
            for i in range(n-2):
                distance += self.finalFormula((i+1)*dx)*dx
            return distance
        
        def Simpson(self,redshift,n):
            dx = redshift/(2*n)
            distance = (self.finalFormula(0)+self.finalFormula(redshift))/3*dx
            for i in range(n):
                distance += (4*self.finalFormula((2*i+1)*dx))/3*dx
            for j in range(n-1):
                distance += (2*self.finalFormula((2*(j+1))*dx))/3*dx
            return distance

        #4.3---New Methods
        def cumulativeTrapezoid(self,redshift,n):
            dx = redshift/(n-1)
            z = [0]
            d = [0]
            for i in range(n-1):
                d.append(d[i]+(self.finalFormula((i)*dx)+self.finalFormula((i+1)*dx))/2*dx)
                z.append((i+1)*dx)
            redshifts = array.array('d',z)
            distances = array.array('d',d)
            return redshifts,distances
        
        def ICD(self,zArray):
        #Input a redshift array into this method to get distance array calculated by interoplation
            z,dArray = self.cumulativeTrapezoid(1,50)
            f = interp1d(np.array(z),np.array(dArray))
            ICDarray=[]
            for z in zArray:
                ICDarray.append(f(z))
            return array.array('d',ICDarray)
        
        def ICM(self,zArray,megaParsecs):
        #Input a redshift array into this method to get moduli array calculated by interpolation
            z,dArray = self.cumulativeTrapezoid(1,50)
            f = interp1d(np.array(z),np.array(dArray))
            ICMarray=[]
            for z in zArray:
                x = self.Omega_k**0.5*self.H0/(3*10**8)*f(z)
                if self.Omega_k > 0:
                   s=np.sinh(x)
                elif self.Omega_k == 0:
                   s=x
                elif self.Omega_k < 0:
                   s=np.sin(x)
                DL = (1+z)*3*10**8/self.H0*1/(abs(self.Omega_k)**0.5)*s
                u = 5*np.log10(DL/megaParsecs)+25
                ICMarray.append(u)
            return array.array('d',ICMarray)



    #Tester Functions
    def plotGraph(H0,Omega_m,Omega_lambda):
        D = []  
        z = []  
        Omega_k = 1-Omega_m-Omega_lambda
        for i in range(100):
           z.append(i/100)
           D.append(ZtoDFormula(H0,Omega_m,Omega_lambda,Omega_k,i/100))
    
        plt.clf()
        plt.plot(z,D)
        plt.title(f'Integrand versus Redshift\n(Omega_m:{Omega_m}, Omega_lambda:{Omega_lambda})')
        plt.xlabel('Redshift')
        plt.ylabel('Intergrand')
        pngName = f'IntegrandPlot({Omega_m},{Omega_lambda}).png'
        plt.savefig(pngName)
        



    #-----Unit3 New Content-----
    # 4.1---Testing and Comment
    newCosmology = Cosmology(72,0.3,0.6)
    """
        Parameter Declaration:
        redshift: Usually from 0-1.
        n: The number of points, no unit.

        Methods:
        .rectangle: Retures the distance(mpc) calculated by rectangle rule.
        .trapezoid: Retures the distance(mpc) calculated by trapezoid rule.
        .Simpson:  Retures the distance(mpc) calculated by Simpson's rule.
    """
    def testing41(cosmology):
        print(cosmology.rectangle(1,50))
        print(cosmology.trapezoid(1,50))
        print(cosmology.Simpson(1,50))
    #testing41(newCosmology)

    # 4.2---High Precise Estimate
    HPDistance = newCosmology.Simpson(1,10000)

    # 4.2---Absolute Fractional Error Plot
    def errorPlot(cosmology,step,startNumber,endNumber):
        errorR = []  
        errorT = []
        errorS = []
        n = []  
        for i in range(math.floor(startNumber/step),math.floor(endNumber/step)):
            n.append(i*step)
            errorR.append(abs(cosmology.rectangle(1,i*step)-HPDistance)/HPDistance)
            errorT.append(abs(cosmology.trapezoid(1,i*step)-HPDistance)/HPDistance)
            errorS.append(abs(cosmology.Simpson(1,i*step)-HPDistance)/HPDistance)
    
        plt.clf()
        plt.plot(n,errorR)
        plt.plot(n,errorT)
        plt.scatter(n,errorS)
        plt.title(f'Absolute Fractional Error versus point number')
        plt.xlabel('Point Number')
        plt.ylabel('Absolute Fractional Error')
        plt.savefig('errorPlot.png')
    #This function gives plot about 3 absolute fractional errors versus step numbers
    #errorPlot(newCosmology,10,10,1000)


    # 4.3---Testing and Comment
    """
        Parameter Declaration:
        redshift: Usually from 0-1.
        n: The number of points, no unit.

        Methods:
        .cumulativeTrapezoid Returns 2 arrays.
        The first one including redshift values at each step, the second one including distances(Mpc) at each step.
    """
    def cTPlot(z,dArray):
        #Testing Plot
        plt.clf()
        plt.plot(z,dArray)
        plt.title(f'Distance versus Redshift')
        plt.xlabel('redshift')
        plt.ylabel('Distance(mpc)')
        plt.savefig('cTPlot.png')
    #cTPlot(newCosmology.cumulativeTrapezoid(1,50))


    # 4.3---Distance and Moduli Calculation by Interpolation
    """
        Parameter Declaration:
        zArray: A redshift array in any order

        Methods:
        .ICD Returns A distances(Mpc) array respectively to each redshift value.
        .ICM Returns A modulus array respectively to each redshift value. 
    """

    # 4.4---Moduli Ploting and Exploration
    zArray = array.array('d',[x*1/100 for x in range(50,100)]) #Testing redshift values

    def ICMPlot():
        plt.clf()
        for i in range(1,7):
            plt.plot(zArray,Cosmology(73,0.05*i,0.6).ICM(zArray,1))
        plt.title(f'Modulus versus Redshift')
        plt.xlabel('Redshift')
        plt.ylabel('Modulus')
        pngName = f'ICMPlotM.png'
        plt.savefig(pngName)
    ICMPlot()



main()
    
          
  
   