
import numpy as np
import array
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from cosmology import Cosmology



# 4.1---Likelihood Class
class Likelihood:
    """
    This Class is for cosmology model calculations. It can contructs an instance basing on
    the input cosmology data. The instance can do four things:

        1. Storing the input cosmology data into arrays; 
        2. Calculating the log likelihood of the data with a given model;
        3. Calculating a moduli array for each redshift value in the data with a given model;
        4. Calculating a residual array for each redshift value in the data with a given model;
    """
    def __init__(self,path):
        """
        Create a likelihood instance.

        Parameters
        ----------
        path : string
            The path of the cosmology data file.
        """
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            dataLines = content.split('\n')
            redshiftList = []
            mobsList = []
            sigmaList = []
            for i in range(len(dataLines)):
                if not dataLines[i].startswith('#') and dataLines[i] != '':
                    datas = dataLines[i].split(' ')
                    redshiftList.append(float(datas[0]))
                    mobsList.append(float(datas[1]))
                    sigmaList.append(float(datas[2]))
        self.redShiftArray = array.array('d',redshiftList)
        self.mobsArray = array.array('d',mobsList)
        self.sigmaArray = array.array('d',sigmaList)
        
    def __call__(self, parameterVector, interpolationNumber):
        """
        Calculate the log likelihood of the data at a given parameter set (model).

        Parameters
        ----------
        parameterVector : array
            An array inculding three parameters (float) in sequence.
            They should be in a form: [Omega_m, Omega_lambda, H0]
        interpolationNumber : int
            The number of points used for interpolation.

        Returns
        -------
        float
            The log likelihood at the given parameter set. Dimensionless.
        """
        cosmology0 = Cosmology(parameterVector[2],parameterVector[0],parameterVector[1])
        moduliArray = cosmology0.ICM(self.redShiftArray,1,interpolationNumber)
        logLikelihood = 0
        for i in range(0,len(self.redShiftArray)):
                logLikelihood += -0.5*(self.mobsArray[i]-(moduliArray[i]-19.3))**2/self.sigmaArray[i]**2
        return logLikelihood
        
    def getModuliArray(self,parameterVector,interpolationNumber):
        """
        Calculate the moduli array of the data at a given parameter set (model).

        Parameters
        ----------
        Same as the __call__

        Returns
        -------
        array
            A moduli array for each redshift value in the data at the given parameter set. Dimensionless.
        """
        cosmology0 = Cosmology(parameterVector[2],parameterVector[0],parameterVector[1])
        moduliArray = cosmology0.ICM(self.redShiftArray,1,interpolationNumber)
        for i in range(0,len(self.redShiftArray)):
            moduliArray[i] = moduliArray[i]-19.3
        return moduliArray
        
    def getResidualArray(self,parameterVector,interpolationNumber):
        """
        Calculate the residual array of the data at a given parameter set (model).

        Parameters
        ----------
        Same as the __call__

        Returns
        -------
        array
            A residual array for each redshift value in the data at the given parameter set. Dimensionless.
        """
        cosmology0 = Cosmology(parameterVector[2],parameterVector[0],parameterVector[1])
        moduliArray = cosmology0.ICM(self.redShiftArray,1,interpolationNumber)
        residualArray = []
        for i in range(0,len(self.redShiftArray)):
            residualArray.append((self.mobsArray[i]-(moduliArray[i]-19.3))/self.sigmaArray[i])
        return array.array('d',residualArray)
    
    

# 4.2---Optimize Likelihood Function  4.3---Keywords Added to Change Model
def optimize(likelihoodInstance,estimatedParameters,interpolationNumber,fixOmegaLambda=False):
    """
        Calculate the max-likelihood optimized parameter set
        and the value of log likelihood at the optimized model.

        Parameters
        ----------
        likelihoodInstance : object
            A likelihood Instance created by the Likelihood class.
        estimatedParameters : array
            An array inculding three parameters (float) in sequence.
            They should be in a form: [Omega_m, Omega_lambda, H0]
        interpolationNumber : int
            The number of points used for interpolation.

        Keywords
        --------
        fixOmegaLambda : boolean
            Set at false in default. If set it at true,
            the value of Omega_lambda will be fixed at 0.

        Returns
        -------
        array
            Optimized parameter set with the given data.
        float
            Value of log likelihood at the optimized model. Dimensionless.
    """
    def targetFunction(x):
        return -likelihoodInstance(x,interpolationNumber)
    notNegative = [(0,np.inf),(0,0),(0,np.inf)] if fixOmegaLambda else [(0,np.inf),(0,np.inf),(0,np.inf)]
    bestParameters = minimize(targetFunction,x0=estimatedParameters,method='L-BFGS-B',bounds=notNegative)
    return bestParameters.x,-bestParameters.fun
    




def main():
    # Constructing an likelihood instance basing on the data.
    likelihood0 = Likelihood('pantheon_data.txt')

    # 4.1---Calculation of likelihood
    p0 = likelihood0(np.array([0.3,0.699,70]),100)
    print(p0)

    # 4.1---Convergence Plot
    def convergencePlot(likelihoodInstance,startNumber,endNumber):
        """
        This function plot the value of log likelihood versus the number of interpolation.
        """
        NValues = []
        pValues = []
        deltaPValues = [0]
        for i in range(startNumber,endNumber,1):
            NValues.append(i)
            pValues.append(likelihoodInstance(np.array([0.3,0.699,70]),i))
            if len(pValues) > 1:
               deltaPValues.append(pValues[len(pValues)-1]-pValues[len(pValues)-2])
        plt.clf()
        plt.plot(NValues,pValues)
        plt.title(f'Log Likelihood versus Number of Interpolation')
        plt.xlabel('Number of Interpolation')
        plt.ylabel('Log Likelihood')
        plt.savefig('covergencePlot.png')

        plt.clf()
        plt.plot(NValues,deltaPValues)
        plt.title(f'Delta Log Likelihood versus Number of Interpolation')
        plt.savefig('deltaPPlot.png')
    convergencePlot(likelihood0,15,500)

    # 4.2---Calculate a optimized parameter set
    interpolationNumber = 100
    estimatedParameters = [0.3,0.699,70]
    bestParameters,maxLogLikelihood = optimize(likelihood0,estimatedParameters,interpolationNumber,fixOmegaLambda=False)
    print(bestParameters[0],bestParameters[1],bestParameters[2],maxLogLikelihood)


    # 4.2---Plot of Fitting
    def fittingPlot():
        """
        This function plot the value of distance moduli versus redshift
        to show how does a model go through the data points.
        """
        moduliArray0 = likelihood0.getModuliArray(estimatedParameters,interpolationNumber)
        moduliArray = likelihood0.getModuliArray(bestParameters,interpolationNumber)
        plt.clf()
        plt.scatter(likelihood0.redShiftArray,likelihood0.mobsArray,s=10,alpha=0.15)
        plt.errorbar(likelihood0.redShiftArray,likelihood0.mobsArray,yerr=likelihood0.sigmaArray,capsize=2,alpha=0.3)
        plt.plot(likelihood0.redShiftArray,moduliArray,c='green',alpha=0.6)
        plt.plot(likelihood0.redShiftArray,moduliArray0,c='red',alpha=0.6)
        plt.title(f'Distance Moduli versus Redshift')
        plt.xlabel('Redshift')
        plt.ylabel('Distance Moduli')
        plt.savefig('fittingPlot.png')
    fittingPlot()


    # 4.2---Plot of Residual
    def residualPlot():
        """
        This function plot the residual of distance moduli versus redshift
        to see whether the model is a good fit to the data.
        """
        residualArray0 = likelihood0.getResidualArray(estimatedParameters,interpolationNumber)
        mean0 = round(np.mean(residualArray0),3)
        std0 = round(np.var(residualArray0)**0.5,3)
        residualArray = likelihood0.getResidualArray(bestParameters,interpolationNumber)
        mean = round(np.mean(residualArray),3)
        std = round(np.var(residualArray)**0.5,3)
        plt.clf()
        plt.scatter(likelihood0.redShiftArray,residualArray,s=10,c='green',alpha=0.15)
        plt.annotate(f'mean:{mean}, std:{std}',[1.5,2.5],c='green')
        plt.scatter(likelihood0.redShiftArray,residualArray0,s=10,c='red',alpha=0.15)
        plt.annotate(f'mean:{mean0}, std:{std0}',[1.5,2.2],c='red')
        plt.title(f'Residual of Distance Moduli versus Redshift')
        plt.xlabel('Redshift')
        plt.ylabel('Residual of Distance Moduli')
        plt.savefig('residualPlot.png')
    residualPlot()

#main()    
            

