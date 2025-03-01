import numpy as np
import array
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from cosmology import Cosmology



# Class Used to Calculate Likelihood (Copied From CP4.py)
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
        moduliArray = cosmology0.ICM(self.redShiftArray,interpolationNumber)
        logLikelihood = 0
        for i in range(0,len(self.redShiftArray)):
                logLikelihood += -0.5*(self.mobsArray[i]-(moduliArray[i]-19.3))**2/self.sigmaArray[i]**2
        return logLikelihood

likelihood0 = Likelihood('pantheon_data.txt')
maxLikelihood = -412

def threeDGrid(startVector,endVector):
    likelihoodI = []
    for i in range(0,10):
        omega_m = startVector[0]+i*(endVector[0]-startVector[0])/9
        likelihoodJ = []
        for j in range(0,10):
            omega_lambda = startVector[1]+j*(endVector[1]-startVector[1])/9
            likelihoodK = []
            for k in range(0,10):
                H0 = startVector[2]+k*(endVector[2]-startVector[2])/9
                likelihoodK.append(likelihood0([omega_m,omega_lambda,H0],100))
            likelihoodJ.append(likelihoodK)
        likelihoodI.append(likelihoodJ)
    return np.array(likelihoodI)

def twoDGrid(startVector,endVector,stepNumberVector):
    parameter1 = []
    parameter2 = []
    likelihoodI = []
    for i in range(0,stepNumberVector[0]+1):
        omega_m = startVector[0]+i*(endVector[0]-startVector[0])/stepNumberVector[0]
        likelihoodJ = []
        for j in range(0,stepNumberVector[1]+1):
            omega_lambda = startVector[1]+j*(endVector[1]-startVector[1])/stepNumberVector[1]
            likelihoodK = 0
            for k in range(0,stepNumberVector[2]+1):
                H0 = startVector[2]+k*(endVector[2]-startVector[2])/stepNumberVector[2]
                likelihoodK += np.exp(likelihood0([omega_m,omega_lambda,H0],100)-maxLikelihood)
            if i == 0: 
                parameter2.append(omega_lambda)
            likelihoodJ.append(np.log(likelihoodK)+maxLikelihood)
        parameter1.append(omega_m)
        likelihoodI.append(likelihoodJ)
    return parameter1,parameter2,likelihoodI

def oneDGrid(startVector,endVector,stepNumberVector):
    parameter = []
    likelihoodI = []
    for i in range(0,stepNumberVector[0]+1):
        omega_lambda = startVector[0]+i*(endVector[0]-startVector[0])/stepNumberVector[0]
        likelihoodJ = 0
        for j in range(0,stepNumberVector[1]+1):
            omega_m = startVector[1]+j*(endVector[1]-startVector[1])/stepNumberVector[1]
            likelihoodK = 0
            for k in range(0,stepNumberVector[2]+1):
                H0 = startVector[2]+k*(endVector[2]-startVector[2])/stepNumberVector[2]
                likelihoodK += np.exp(likelihood0([omega_m,omega_lambda,H0],100)-maxLikelihood)
            likelihoodJ += likelihoodK
        parameter.append(omega_lambda)
        likelihoodI.append(np.log(likelihoodJ)+maxLikelihood)
    return parameter,likelihoodI

def plotTwoDGrid(parameter1,parameter2,logLikelihood):
    plt.imshow(logLikelihood, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.ylabel('Omega_m')
    plt.xlabel('Omega_k')
    y_positions = np.linspace(0,len(parameter1)-1,len(parameter1))
    x_positions = np.linspace(0,len(parameter2)-1,len(parameter2))
    plt.yticks(y_positions,labels=[f"{y}" for y in parameter1])
    plt.xticks(x_positions,labels=[f"{x}" for x in parameter2],rotation=-90)
    plt.title(f'Marginalized Log Likelihood versus Omega_m and Omega_k')
    plt.savefig('twoDPlot_H0.png')

def plotOneDGrid(parameter,logLikelihood):
    plt.clf()
    plt.plot(parameter,logLikelihood)
    plt.title(f'Marginalized Log Likelihood versus Omega_lambda')
    plt.xlabel('Omega_lambda')
    plt.ylabel('Marginalized Log Likelihood')
    plt.savefig('oneDPlot_Omega_lambda.png')

#parameter0,oneDLikelihood = oneDGrid([0,0,60],[1,1,80],[100,10,10])
#plotOneDGrid(parameter0,oneDLikelihood)

#parameter1,parameter2,twoDLikelihood = twoDGrid([0,0,60],[1,1,80],[5,5,5])
#plotTwoDGrid(parameter1,parameter2,twoDLikelihood)

class Metropolis:
    def __init__(self,logLikelihoodFunction):
        self.L = logLikelihoodFunction
    
    def run(self,startVector,stepVector,numberOfPoints):
        p = []
        likelihood = []
        p.append(startVector)
        likelihood.append(self.L(startVector,100))
        i = 0
        numberOfTimes = 0
        while i < numberOfPoints:
            numberOfTimes += 1
            newStepRandom = np.random.normal(loc=0,scale=1,size=len(stepVector)).tolist()
            newVector = []
            for j in range(0,len(stepVector)):
                newVector.append(p[len(p)-1][j]+newStepRandom[j]*stepVector[j])
            newLikelihood = self.L(newVector,100)
            if newLikelihood >= likelihood[len(likelihood)-1]:
                p.append(newVector)
                likelihood.append(newLikelihood)
                i += 1
            else:
                u = np.random.uniform(0,1)
                if np.log(u) < newLikelihood-likelihood[len(likelihood)-1]:
                    p.append(newVector)
                    likelihood.append(newLikelihood)
                    i += 1
        print(numberOfTimes)
        return p,likelihood
            
metropolis0 = Metropolis(likelihood0)
vectors,logLikelihood = metropolis0.run([0.3,0.7,70],[0.07,0.1,0.3],50)

def threeDScatter






