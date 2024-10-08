
import numpy as np

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
    matrixM = task4a(5) #Test Matrix
    #print (task4b(3,matrixM))

main()

   