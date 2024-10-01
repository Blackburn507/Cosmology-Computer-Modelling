
import numpy
import matplotlib.pyplot as plt
from scipy.integrate import quad

#test

def main():
    #Formula
    def ZtoDFormula(Omega_m,Omega_lambda,Omega_k,x):
        return (Omega_m*(1+x)**3+Omega_k*(1+x)**2+Omega_lambda)**(-0.5)   


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
    
        def calculateD(self,redshift):
            c = 3*10**8
            result,error = quad(ZtoDFormula,0,redshift,args=(self.Omega_m,self.Omega_lambda,self.Omega_k,))
            return c*result/self.H0
        

    #Tester Functions
    def plotGraph(Omega_m,Omega_lambda):
        D = []  
        z = []  
        Omega_k = 1-Omega_m-Omega_lambda
        for i in range(100):
           z.append(i/100)
           D.append(ZtoDFormula(Omega_m,Omega_lambda,Omega_k,z[i]))
    
        plt.clf()
        plt.plot(z,D)
        plt.title(f'Integrand versus Redshift\n(Omega_m:{Omega_m}, Omega_lambda:{Omega_lambda})')
        plt.xlabel('Redshift')
        plt.ylabel('Intergrand')
        pngName = f'IntegrandPlot({Omega_m},{Omega_lambda}).png'
        plt.savefig(pngName)
        plt.savefig('test.png')
        
    
    #Objects
    def CosmologyTest(H0,Omega_m,Omega_lambda):
        newCosmology = Cosmology(H0,Omega_m,Omega_lambda)
        cosmologyInfo = f"Distance: {newCosmology.calculateD(1)}, ifFlat: {newCosmology.ifFlat}, Omega_m*h^2: {newCosmology.Omega_m_h_square}"
        print(newCosmology.modelDescribe)
        print(cosmologyInfo)
        plotGraph(Omega_m,Omega_lambda)

    CosmologyTest(67.8,0.3,0.7)
    CosmologyTest(67.8,1,0.7)
    CosmologyTest(67.8,3,0.7)

    #Comment
    #The Intergrand becomes more curved when the Omega_m becomes larger.

main()
    
          
