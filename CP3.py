


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
    #print(newCosmology.rectangle(1,50))
    #print(newCosmology.trapezoid(1,50))
    #print(newCosmology.Simpson(1,50))
    

    # 4.2---High Precise Estimate
    HPDistance = newCosmology.Simpson(1,10000)
    #with open("documentText",'w',encoding='utf-8') as file:
        #file.write(str(HPDistance))

    # 4.2---Absolute Fractional Error Plot
    def errorPlot(step,startNumber,endNumber):
        errorR = []  
        errorT = []
        errorS = []
        n = []  
        for i in range(math.floor(startNumber/step),math.floor(endNumber/step)):
           n.append(i*step)
           errorR.append(abs(newCosmology.rectangle(1,i*step)-HPDistance)/HPDistance)
           errorT.append(abs(newCosmology.trapezoid(1,i*step)-HPDistance)/HPDistance)
           errorS.append(abs(newCosmology.Simpson(1,i*step)-HPDistance)/HPDistance)
    
        plt.clf()
        #plt.plot(n,errorR)
        #plt.plot(n,errorT)
        #plt.plot(n,errorS)
        plt.title(f'Error of Simpson Estimation versus point number')
        plt.xlabel('Point Number')
        plt.ylabel('Absolute Fractional Error')
        plt.savefig('errorPlotS.png')
    
    #errorPlot(1,10,100)

    # 4.3---Cumulative Trapezoid and Testing
    # The CT method in Cosmology Class outputs 2 lists.
    # The first one is redshift values, the second one is distance values.
    def cTPlot(z,dArray):
        plt.clf()
        plt.plot(z,dArray)
        plt.title(f'Distance versus Redshift')
        plt.xlabel('redshift')
        plt.ylabel('Distance(mpc)')
        plt.savefig('cTPlot.png')
    
    z,dArray = newCosmology.cumulativeTrapezoid(1,50)
    cTPlot(z,dArray)

    # 4.3---Interpolator and Testing
    f = interp1d(np.array(z),np.array(dArray)) #Function Created by Interpolation
    def ICD(zArray):
        ICDarray=[]
        for z in zArray:
            ICDarray.append(f(z))
        return array.array('d',ICDarray)
    
    def ICDPlot(zArray,z,dArray):
        plt.clf()
        plt.plot(z,dArray)
        plt.scatter(zArray,ICD(zArray),s=10)
        plt.title(f'Interpolated Distance versus Redshift')
        plt.xlabel('redshift')
        plt.ylabel('Distance(mpc)')
        plt.savefig('ICDPlot.png')

    zArray = array.array('d',[x*1/100 for x in range(1,100)])
    ICDPlot(zArray,z,dArray)

    # 4.3---Moduli and Testing
    Omega_k = newCosmology.Omega_k
    Omega_m = newCosmology.Omega_m
    Omega_lambda = newCosmology.Omega_lambda
    H0 = newCosmology.H0

    def ICM(zArray,Omega_k,H0,megaParsecs):
        ICMarray=[]
        for z in zArray:
            x = Omega_k**0.5*H0/(3*10**8)*f(z)
            if Omega_k > 0:
                s=np.sinh(x)
            elif Omega_k == 0:
                s=x
            elif Omega_k < 0:
                s=np.sin(x)
            DL = (1+z)*3*10**8/H0*1/(abs(Omega_k)**0.5)*s
            u = 5*np.log10(DL/megaParsecs)+25
            ICMarray.append(u)
        return array.array('d',ICMarray)

    # 4.4---Ploting and Exploration
    def ICMPlot(zArray,ICMArray):
        plt.clf()
        plt.plot(zArray,ICMArray)
        plt.title(f'Moduli versus Redshift (H0:{H0},Omega_m:{Omega_m}, Omega_lambda:{Omega_lambda})')
        plt.xlabel('redshift')
        plt.ylabel('Moduli')
        pngName = f'ICMPlot({H0},{Omega_m},{Omega_lambda}).png'
        plt.savefig(pngName)

    ICMPlot(zArray,ICM(zArray,Omega_k,H0,1))



main()
    
          
  
   