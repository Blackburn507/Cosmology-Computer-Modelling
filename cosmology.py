
import numpy as np
import array
from scipy.interpolate import interp1d

#Class and Methods
class Cosmology:
        def __init__(self,H0,Omega_m,Omega_lambda):
            self.H0 = H0
            self.Omega_m = Omega_m
            self.Omega_lambda = Omega_lambda
            self.Omega_k = 1-Omega_m-Omega_lambda
            self.lightspeed = 2.9979*10**5
            self.megaParsec = 1

        def finalFormula(self,x):
            return (self.lightspeed)/self.H0*(self.Omega_m*(1+x)**3+self.Omega_k*(1+x)**2+self.Omega_lambda)**(-0.5)   

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
        
        def ICM(self,zArray,N):
        #Input a redshift array into this method to get moduli array calculated by interpolation
        #N is the interpolation number
            z,dArray = self.cumulativeTrapezoid(2.3,N)
            f = interp1d(np.array(z),np.array(dArray))
            ICMarray=[]
            for z in zArray:
                DL = 0
                x = abs(self.Omega_k)**0.5*self.H0/(self.lightspeed)*f(z)
                if self.Omega_k > 0:
                   DL = (1+z)*self.lightspeed/self.H0*1/(abs(self.Omega_k)**0.5)*np.sinh(x)
                elif self.Omega_k == 0:
                   DL = (1+z)*f(z)
                elif self.Omega_k < 0:
                   DL = (1+z)*self.lightspeed/self.H0*1/(abs(self.Omega_k)**0.5)*np.sin(x)
                u = 5*np.log10(DL/self.megaParsec)+25
                ICMarray.append(u)
            return array.array('d',ICMarray)

"""        
cosmology0 = Cosmology(70,0.3,0.7)
moduliArray = cosmology0.ICM([0.5,1,1.5],100)
print(moduliArray)
"""
