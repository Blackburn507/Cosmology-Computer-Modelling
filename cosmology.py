
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
            self.Omega_m_h_square = Omega_m*(H0/100)**2
            self.ifFlat = "true" if self.Omega_k==0 else "false"
            self.modelDescribe = f"Cosmology with H0={H0}, Omega_m={Omega_m}, Omega_lambda={Omega_lambda}, Omega_k={self.Omega_k}"

        def finalFormula(self,x):
            return (3*10**5)/self.H0*(self.Omega_m*(1+x)**3+self.Omega_k*(1+x)**2+self.Omega_lambda)**(-0.5)   

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
        
        def ICM(self,zArray,megaParsecs,N):
        #Input a redshift array into this method to get moduli array calculated by interpolation
        #N is the interpolation number
            z,dArray = self.cumulativeTrapezoid(2.3,N)
            f = interp1d(np.array(z),np.array(dArray))
            ICMarray=[]
            for z in zArray:
                x = abs(self.Omega_k)**0.5*self.H0/(3*10**8)*f(z)
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
