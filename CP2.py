
import numpy as np
import os

def main():
    def task1a(m,n):
        """
        This function output 3 arrays respectively as the Task1a asked
        """
        array1 = [0]*m
        array2 = [i+1 for i in range(n)]
        array3 = np.random.rand(m,n)
        return array1,array2,array3
    #print(task1a(5,5))

    def task1c(m,n):
        _,array2,array3 = task1a(m,n)
        meanArray2 = np.mean(array2)
        maxArray3 = np.max(array3)
        return meanArray2,maxArray3
    #print(task1c(5,5))
    
    def task1d(arrayA):
        for i in range(len(arrayA)):
          arrayA[i] = arrayA[i]**2
    arrayA = [1,2,3,4,5]
    task1d(arrayA)
    #print(arrayA)

    def task2a(t,vectorA,vectorB):
        return t*(vectorA+vectorB)
    #print (task2a(10,np.array([1,1,4,5,1,4]),np.array([1,9,1,9,8,1])))

    def task2b(vectorX,vectorY):
        return np.linalg.norm(vectorX-vectorY)
    #print (task2b(np.array([1,1,4]),np.array([5,1,4])))
    
    def task3a(vector1,vector2):
        LHS = np.cross(vector1,vector2)
        RHS = -1*np.cross(vector2,vector1)
        return LHS,RHS
    #print (task3a(np.array([1,1,4]),np.array([5,1,4])))

    def task3b(vector1,vector2,vector3):
        LHS = np.cross(vector1,vector2+vector3)
        RHS = np.cross(vector1,vector2)+np.cross(vector1,vector3)
        return LHS,RHS
    #print (task3b(np.array([1,1,4]),np.array([5,1,4]),np.array([1,9,1])))

    def task3c(vector1,vector2,vector3):
        LHS = np.cross(vector1,np.cross(vector2,vector3))
        RHS = np.dot(vector1,vector3)*vector2-np.dot(vector1,vector2)*vector3
        return LHS,RHS
    #print (task3c(np.array([1,1,4]),np.array([5,1,4]),np.array([1,9,1])))

    def task4a(n):
        matrixM = np.fromfunction(lambda i,j:i+2*j,(n,n),dtype=int)
        return matrixM
    #print (task4a(5))

    def task4b(n,matrixM):
        for i in range(len(matrixM)):
            for j in range(n):
                arrayA[i]+=matrixM[i][j]  
        return arrayA
    matrixM = task4a(5) #Tester Matrix
    #print (task4b(3,matrixM))

    def task5(d,u,o):
        sum = 0
        for i in range(len(d)):
            sum += ((d[i]-u[i])/o[i])**2
        sum *= 0.5
        return sum/2
    d = np.array([1,4,2,8,5,7])
    u = np.array([1,1,4,5,1,4])
    o = np.array([1,2,3,4,5,6])
    #print(task5(d,u,o))

    def task6(path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            dataLines = content.split('\n')
            omegamh2 = []
            omegabh2 = []
            H0 = []
            for i in range(len(dataLines)):
                if not dataLines[i].startswith('#') and dataLines[i] != '':
                    datas = dataLines[i].split(' ')
                    omegamh2.append(float(datas[0]))
                    omegabh2.append(float(datas[1]))
                    H0.append(float(datas[2]))
            mean = []
            variance = []
            mean.append(np.mean(omegamh2))
            mean.append(np.mean(omegabh2))
            mean.append(np.mean(H0))
            variance.append(np.var(omegamh2))
            variance.append(np.var(omegabh2))
            variance.append(np.var(H0))
            return mean,variance
    #print (task6('data.txt'))
    #Input the path of data.txt into this function. It outputs mean and variance for each data column.

main()

   