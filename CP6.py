import numpy as np
import array
import matplotlib.pyplot as plt
import time
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



# Likelihood Gird Calculation (Improved From CP5.py)
def threeDGrid(likelihoodInstance,startVector,endVector,stepNumberVector):
    """
    This function calculate a 3D log likelihood grid at the range given. 
    The step number on each parameter is custom.
    Parameters: callable object,array,array,array
    Returns: array
    """
    H0Array = []
    Omega_lambdaArray = []
    Omega_mArray = []
    likelihoodI = []
    for i in range(0,stepNumberVector[0]+1):
        H0 = startVector[0]+i*(endVector[0]-startVector[0])/stepNumberVector[0]
        likelihoodJ = []
        for j in range(0,stepNumberVector[1]+1):
            omega_lambda = startVector[1]+j*(endVector[1]-startVector[1])/stepNumberVector[1]
            likelihoodK = []
            for k in range(0,stepNumberVector[2]+1):
                omega_m = startVector[2]+k*(endVector[2]-startVector[2])/stepNumberVector[2]
                if i == 0 and j == 0:
                    Omega_mArray.append(omega_m)
                likelihoodK.append(likelihoodInstance([omega_m,omega_lambda,H0],100))
            if i == 0: 
                Omega_lambdaArray.append(omega_lambda)
            likelihoodJ.append(likelihoodK)
        H0Array.append(H0)
        likelihoodI.append(likelihoodJ)
    threeDLikelihood = np.array(likelihoodI)
    max_likelihood = np.max(threeDLikelihood)
    tDLN = np.exp(threeDLikelihood - max_likelihood)
    return np.array(H0Array),np.array(Omega_lambdaArray),np.array(Omega_mArray),tDLN

def marginalization(tDL,startVector,endVector,stepNumberVector):
    """
    This function do both marginalization and normalization for the imput 3D grid.
    """
    TD1 = np.sum(tDL, axis=0)/np.sum(tDL)/((endVector[1]-startVector[1])/stepNumberVector[1])/((endVector[2]-startVector[2])/stepNumberVector[2])
    TD2 = np.sum(tDL, axis=1)/np.sum(tDL)/((endVector[0]-startVector[0])/stepNumberVector[0])/((endVector[2]-startVector[2])/stepNumberVector[2])
    TD3 = np.sum(tDL, axis=2)/np.sum(tDL)/((endVector[0]-startVector[0])/stepNumberVector[0])/((endVector[1]-startVector[1])/stepNumberVector[1])
    OD1 = np.sum(tDL, axis=(1,2))/np.sum(tDL)/((endVector[0]-startVector[0])/stepNumberVector[0])
    OD2 = np.sum(tDL, axis=(0,2))/np.sum(tDL)/((endVector[1]-startVector[1])/stepNumberVector[1])
    OD3 = np.sum(tDL, axis=(0,1))/np.sum(tDL)/((endVector[2]-startVector[2])/stepNumberVector[2])
    return TD1,TD2,TD3,OD1,OD2,OD3

def plotTwoDGrid(H0Array,Omega_lambdaArray,Omega_mArray,TD1,TD2,TD3):
    plt.clf()
    plt.imshow(TD1, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.ylabel('Omega_lambda')
    plt.xlabel('Omega_m')
    y_positions = np.linspace(0,len(Omega_lambdaArray)-1,9)
    x_positions = np.linspace(0,len(Omega_mArray)-1,7)
    plt.yticks(y_positions,labels=[f"{round(i*0.1,1)}" for i in range(0,9)])
    plt.xticks(x_positions,labels=[f"{round(i*0.1,1)}" for i in range(0,7)])
    plt.title(f'Marginalized 2D Likelihood versus Omega_lambda and Omega_m')
    plt.savefig('twoDPlot_H0.png')

    plt.clf()
    plt.imshow(TD2, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.ylabel('H0')
    plt.xlabel('Omega_m')
    y_positions = np.linspace(0,len(H0Array)-1,11)
    x_positions = np.linspace(0,len(Omega_mArray)-1,7)
    plt.yticks(y_positions,labels=[f"{round(69+i*0.3,1)}" for i in range(0,11)])
    plt.xticks(x_positions,labels=[f"{round(i*0.1,1)}" for i in range(0,7)])
    plt.title(f'Marginalized 2D Likelihood versus H0 and Omega_m')
    plt.savefig('twoDPlot_Omega_lambda.png')

    plt.clf()
    plt.imshow(TD3, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.ylabel('H0')
    plt.xlabel('Omega_lambda')
    y_positions = np.linspace(0,len(H0Array)-1,11)
    x_positions = np.linspace(0,len(Omega_lambdaArray)-1,9)
    plt.yticks(y_positions,labels=[f"{round(69+i*0.3,1)}" for i in range(0,11)])
    plt.xticks(x_positions,labels=[f"{round(i*0.1,1)}" for i in range(0,9)])
    plt.title(f'Marginalized 2D Likelihood versus H0 and Omega_lambda')
    plt.savefig('twoDPlot_Omega_m.png')

def plotOneDGrid(H0Array,Omega_lambdaArray,Omega_mArray,OD1,OD2,OD3):
    plt.clf()
    plt.plot(H0Array,OD1,c='red')
    plt.title(f'Marginalized 1D Likelihood versus H0')
    plt.xlabel('H0')
    plt.ylabel('Marginalized Likelihood')
    plt.savefig('oneDPlot_H0.png')

    plt.clf()
    plt.plot(Omega_lambdaArray,OD2,c='red')
    plt.title(f'Marginalized 1D Likelihood versus Omega_lambda')
    plt.xlabel('Omega_lambda')
    plt.ylabel('Marginalized Likelihood')
    plt.savefig('oneDPlot_Omega_lambda.png')

    plt.clf()
    plt.plot(Omega_mArray,OD3,c='red')
    plt.title(f'Marginalized 1D Likelihood versus Omega_m')
    plt.xlabel('Omega_m')
    plt.ylabel('Marginalized Likelihood')
    plt.savefig('oneDPlot_Omega_m.png')



def testFunctionForGrid(likelihoodInstance):
    startVector = [69,0,0]
    endVector = [72,0.8,0.6]
    stepNumberVector = [30,30,30]

    H0Array,Omega_lambdaArray,Omega_mArray,TRD = threeDGrid(likelihoodInstance,startVector,endVector,stepNumberVector)
    TD1,TD2,TD3,OD1,OD2,OD3 = marginalization(TRD,startVector,endVector,stepNumberVector)
    np.savez('resultsG1.npy',ha=H0Array,ola=Omega_lambdaArray,oma=Omega_mArray,trd=TRD)
    plotTwoDGrid(H0Array,Omega_lambdaArray,Omega_mArray,TD1,TD2,TD3)
    plotOneDGrid(H0Array,Omega_lambdaArray,Omega_mArray,OD1,OD2,OD3)

    maxParametersIndex = np.unravel_index(np.argmax(TRD),TRD.shape)
    maxH0 = startVector[0]+maxParametersIndex[0]*((endVector[0]-startVector[0])/stepNumberVector[0])
    maxOmega_lambda = startVector[1]+maxParametersIndex[1]*((endVector[1]-startVector[1])/stepNumberVector[1])
    maxOmega_m = startVector[2]+maxParametersIndex[2]*((endVector[2]-startVector[2])/stepNumberVector[2])
    meanH0 = np.dot(OD1,H0Array)*((endVector[0]-startVector[0])/stepNumberVector[0])
    meanOmega_lambda = np.dot(OD2,Omega_lambdaArray)*((endVector[1]-startVector[1])/stepNumberVector[1])
    meanOmega_m = np.dot(OD3,Omega_mArray)*((endVector[2]-startVector[2])/stepNumberVector[2])
    varianceH0 = np.dot(OD1,H0Array**2)*((endVector[0]-startVector[0])/stepNumberVector[0])-meanH0**2
    varianceOmega_lambda = np.dot(OD2,Omega_lambdaArray**2)*((endVector[1]-startVector[1])/stepNumberVector[1])-meanOmega_lambda**2
    varianceOmega_m = np.dot(OD3,Omega_mArray**2)*((endVector[2]-startVector[2])/stepNumberVector[2])-meanOmega_m**2

    print('Max = ',maxH0,maxOmega_lambda,maxOmega_m)
    print('Mean = ',meanH0,meanOmega_lambda,meanOmega_m)
    print('Variance = ',varianceH0,varianceOmega_lambda,varianceOmega_m)
    print('Sd = ',np.sqrt(varianceH0),np.sqrt(varianceOmega_lambda),np.sqrt(varianceOmega_m))

L0 = Likelihood('pantheon_data.txt')
#testFunctionForGrid(L0)






# Metropolis (From CP5.py)
# (Mini Project E --- Acceptance Rate Added)
class Metropolis:
    """
    This class is for metropolis analysis of likelihood.
    It has one methods to run the algorithm.
    """
    def __init__(self,logLikelihoodFunction):
        """
        Creates a metropolis instance.

        Parameters
        ----------
        logLikelihoodFunction : function
            An arbitary likelihood function
        """
        self.L = logLikelihoodFunction
    
    def run(self,startVector,stepVector,numberOfPoints):
        """
        This method runs a metropolis algorithm to explore vectors in the parameter space.
        Basing on the likelihood function.

        Parameters
        ----------
        startVector : array
            This is the start vector in the space. (p_0)
            The length of this array should be consistent to the number of parameters of the likelihood function.
        stepVector : array
            This is the step vector.
            The real step vector will basing on this vector with independent gaussian random factors applied on each direction.
        numberOfPoints : int
            The algorithm will keep running until the number of vector it collected meet this value.

        Returns
        -------
        array
            A array of vectors in the parameter space explored by metropolis.
        array
            A array of likelihoods corresponding to the vector explored.
        float
            The acceptance rate of the metropolis chain.
        """
        p = []
        likelihood = []
        p.append(startVector)
        likelihood.append(self.L(startVector,100))
        i = 0
        numberOfTimes = 0
        while i < numberOfPoints:
            numberOfTimes += 1
            newStepRandom = np.array(np.random.normal(loc=0,scale=1,size=len(stepVector)).tolist())
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
        print('Metropolis Completed: ',numberOfTimes)
        return np.array(p),np.array(likelihood),len(p)/numberOfTimes
            
def plotThreeDScatter(vectors):
    parameters = vectors.T
    plt.clf()
    plt.scatter(parameters[0],parameters[1],c=parameters[2],cmap='viridis',s=10,alpha=0.5)
    plt.colorbar(label='H0')
    plt.title(f'Distribution of Points Explored by Metropolis')
    plt.xlabel('Omega_m')
    plt.ylabel('Omega_lambda')
    plt.savefig('metropolisScatter.png')

def plotHist(values,step):
    name = ['Omega_m','Omega_lambda','H0']
    plt.clf()
    plt.hist(values, bins=60, density=True, alpha=0.75, color='black')
    plt.xlabel(f'{name[step]}')
    plt.ylabel('Frequency')
    plt.savefig(f'metropolisHist_{step}.png')

def testFunctionForMetropolis(MetropolisInstance):
    vectors,logLikelihood,acceptanceRate = MetropolisInstance.run([0.3,0.45,70],[0.07,0.1,0.3],10000)
    parameters = vectors.T
    np.save('resultsM.npy',parameters)
    maxLikelihoodIndex = np.unravel_index(np.argmax(logLikelihood),logLikelihood.shape)
    plotThreeDScatter(vectors)
    plotHist(parameters[0],0)
    plotHist(parameters[1],1)
    plotHist(parameters[2],2)
    print('Acceptance Rate = ',acceptanceRate)
    print(parameters[0][maxLikelihoodIndex],parameters[1][maxLikelihoodIndex],parameters[2][maxLikelihoodIndex])
    print(np.mean(parameters[0]),np.mean(parameters[1]),np.mean(parameters[2]))
    print(np.var(parameters[0]),np.var(parameters[1]),np.var(parameters[2]))

def convergencePlotForMetropolis(MetropolisInstance):
    meanArray = []
    varianceArray = []
    aR = []
    chainLength = [10,20,30,50,70,100,150,200,250,300,400,500,600,700,800,1000]
    for i in range(0,len(chainLength)):
        vectors,logLikelihood,acceptanceRate = MetropolisInstance.run([0.3,0.7,70],[0.07,0.1,0.3],chainLength[i])
        parameters = vectors.T
        meanEachParameter = []
        varianceEachParameter = []
        for j in range(0,2):
            meanEachParameter.append(np.mean(parameters[j]))
            varianceEachParameter.append(np.var(parameters[j]))
        meanArray.append(meanEachParameter)
        varianceArray.append(varianceEachParameter)
        aR.append(acceptanceRate)

    plt.clf()
    plt.scatter(chainLength,np.array(meanArray).T[0],c='black',s=10)
    plt.scatter(chainLength,np.array(meanArray).T[1],c='red',s=10)
    plt.savefig('convergence_Mean.png')

    plt.clf()
    plt.scatter(chainLength,np.array(varianceArray).T[0],c='black',s=10)
    plt.scatter(chainLength,np.array(varianceArray).T[1],c='red',s=10)
    plt.savefig('convergence_Variance.png')

    plt.clf()
    plt.scatter(chainLength,np.array(aR),c='red',s=10)
    plt.savefig('convergence_aR.png')

L0 = Likelihood('pantheon_data.txt')
#testFunctionForMetropolis(Metropolis(L0))
#convergencePlotForMetropolis(Metropolis(L0))
#data = np.load('resultsM.npy')


# Mini Project E --- Metropolis Diagnostic
def GRStatistic(MetropolisInstance,startVector,stepVector,M,N):
    """
    This function runs a metropolis algorithm for multiple times to calculate Gelman-Rubin statistic.
        Parameters
        ----------
        MetropolisInstance : object
            A instance created by Metropolis class.
        startVector : array
            This is the start vector in the space of each chain. (p_0)
        stepVector : array
            This is the step vector of each chain.
        M : int
            Number of Metropolis chains.
        N : int
            The length of each Metrpolis chain.

        Returns
        -------
        array
            A array contains values of Gelman-Rubin statistic obtained for each parameter.
    """
    meanArray = []
    varianceArray = []
    for i in range(0,M):
        vectors,logLikelihood,acceptanceRate = MetropolisInstance.run(startVector,stepVector,N)
        parameters = vectors.T
        meanEachParameter = []
        varianceEachParameter = []
        for j in range(0,len(startVector)):
            meanEachParameter.append(np.mean(parameters[j]))
            varianceEachParameter.append(np.var(parameters[j]))
        meanArray.append(meanEachParameter)
        varianceArray.append(varianceEachParameter)
    R = []
    for i in range(0,len(startVector)):
        R.append(np.sqrt((N-1)/N+(M+1)/M*(np.var(np.array(meanArray).T[i])/np.mean(np.array(varianceArray).T[i]))))
    return np.array(R)

def main():
    """
    For test and plot result obtained by GR statistics.
    """
    chainLength = [10,20,30,50,100,150,200,250,300,350,400,450,500]
    RArray = []
    for i in range(0,len(chainLength)):
        RArray.append(GRStatistic(Metropolis(L0),[0.3,0.7,70],[0.07,0.1,0.3],3,chainLength[i]))
    RForOmegaM = np.array(RArray).T[0]-1
    RForOmegaLambda = np.array(RArray).T[1]-1
    RForH0 = np.array(RArray).T[2]-1
    plt.clf()
    plt.scatter(chainLength,np.array(RForOmegaM),c='red',s=10)
    plt.scatter(chainLength,np.array(RForOmegaLambda),c='black',s=10)
    plt.scatter(chainLength,np.array(RForH0),c='blue',s=10)
    plt.savefig('convergence_GR.png')
    print(RArray)

#main()


    




