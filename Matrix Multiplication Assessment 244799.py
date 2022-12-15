#!/usr/bin/env python
# coding: utf-8

# ##         Matrix Multiplication Methods in Python: Comparitive Run-Times and Memory

# #### Candidate Number: 244799

# ### Introduction

# Matrix multiplication is a key concept in linear algebra with many applications including use in machine learning, geometry and network theory. Thus, finding efficient and effective ways of computing matrix multiplications via algorithmic functions is hugely important in the fields of data science, mathematics and computer science. 
# 
# In this report two different methods of computating matrix multiplication in Python are explored and examined; a 'naive' approach which iterates over rows and columns of two matrices respectively, and a method of sub-matrix multiplication and recursion called the Strassen Algorithm, named after it's developer the German mathematician Volker Strassen. The complexity, run times, approximate memory usage with varying sizes of matrices, as well as the limitations of both algorithms will be explored and discussed.
# 
# The first block of code will load the relevant python packages and create some useful functions that will be used throughout the report in the analysis.

# In[62]:


'''
Stephanie L Mathias. October 2021. 
This script explores the different methods of matrix multiplication in Python and their impact on run time.
'''


import numpy as np
import matplotlib.pyplot as plt
import random
import time

#Function to check if an input is a matrix
def IsMatrix(Matrix):
    Length=len(Matrix[0])
    if Length>0:
        for x in Matrix:
            if len(x)==Length:
                continue
            else:
                print("Invalid input.")
    else:
        print("Invalid input.")

#Function that produces summary statistics on the runtimes of a function on given input(s)
def TimeIt(Func,*args,Repeats=5,**kwargs):
    Times=[]
  
    while Repeats>0:
        StartTime=time.time()
        Ans=Func(*args,**kwargs)
        EndTime=time.time()
        TotalTime=EndTime-StartTime
        Times.append(TotalTime)
        Repeats-=1
    
    Mean=np.mean(Times)
    Std=np.std(Times)
    Error=Std/(len(Times)**0.5)
 
    return (Mean,Error,Std)


# ### Matrix Multiplication: Iterative Approach

# First, a function is created to compute matrix multiplication using a 'naive' approach. It takes two matrices as input, checks that the two matrices are dimensionally compatible for multiplication and produces an output by iterating over the rows in the first matrix and the columns of the second matrix, and taking the sum of the products of the elements of the rows and columns of these respectively.

# In[63]:


def MatxMultipl(M1,M2):
    IsMatrix(M1)
    IsMatrix(M2)
    
    m1=len(M1)
    n1=len(M1[0])
    m2=len(M2)
    n2=len(M2[0])
    
    #Check if matrix multiplication is possible
    if m1!=n2:
        print("Matrix multiplication not possible with given inputs.")
        quit()
    else:
        #If compatible, iteratively take the sums of the products of rows of M1 by columns of M2
        ResultMatrix=[[0 for col in range(m1)] for row in range(n2)]
        for i in range(0,m1):
            for j in range(0,m2):
                for k in range(0,n2):
                    ResultMatrix[i][k]+=M1[i][j]*M2[k][j]
        return ResultMatrix


# Next, this function is tested to see how long it takes to run on different sized matrices. The function applied to square matrices (*nxn*) from where *n* = 100 to 300 at intervals of 10, and with elements of integers between 0 and 9.
# For each set of dimensions, the mean of the time taken for 5 runs is stored (as seen in the *TimeIt( )* function above).

# In[64]:


#Run the function and time how long it takes
#NB: this takes quite a few minutes to run
Dimensions=[x for x in range(10,300,10)]
MeanRunTimes=[]

for i in Dimensions:
    Matrix1,Matrix2=np.random.randint(9,size=(i,i)),np.random.randint(9,size=(i,i))
    RunStats=TimeIt(MatxMultipl,Matrix1,Matrix2)
    MeanRunTimes.append(RunStats[0])


# In[65]:


#Here we will plot the our x variable: matrix size against our y variable: mean run time (seconds)
plt.subplots(1,2, figsize=(7,3))
plt.suptitle("Average Run Time of Naive Matrix Multiplication against Matrix Size", fontsize=12)

#Plot raw data
plt.subplot(1,2,1)
plt.scatter(Dimensions,MeanRunTimes,s=20)
plt.xlabel("Dimension of Square Matrix")
plt.ylabel("Mean Time (Seconds)")

#Calculate and plot log10 of raw data
plt.subplot(1,2,2)
LogDimensions=np.log10(Dimensions)
LogMeanRunTimes=np.log10(MeanRunTimes)

plt.scatter(LogDimensions,LogMeanRunTimes,s=20,c='darkgrey')
plt.xlabel("log10 Dimension of Square Matrix")
plt.ylabel("Log10 Mean Time (Seconds)")
plt.tight_layout()
plt.show()


# Above shows the increasing mean run times associated with increased matrix size. Using the log base 10 of the two inputs, we can calculate the gradient to give us an approximate order to run time.

# In[66]:


import scipy.stats as stats

#Use the stats library to calculate the slope
slope,intercept,r_value,p_value,std_err = stats.linregress(LogDimensions,LogMeanRunTimes)
print("The gradient is:\n {} ".format(slope))


# From this, we can deduce that the order of run time for this method of matrix multiplication is approximately $O(n^{3})$

# #### Memory Complexity: Iterative Approach

# The default storage taken by a single integer or float value in Python is 4 bytes each.<sup>1</sup>. This value will be used to approximate the total memory needed when using the iterative multiplication function for square matrices.<br/>
# 
# To calculate the memory required for a square matrices (<i>n</i> by <i>n</i>), we consider the following:
# <br/>1) 4 bytes of memory for every element in the both original matrices to be multiploed: $2 * n^{2}$
# <br/>2) 4 bytes of memory for the resultant matrix, also size <i>n</i> by <i>n</i>: $n^{2}$
# <br/>3) And additional 4 bytes of memory for the object holding the product of two elements within the nester 'for' loop, which is then added to the corresponding element in the resulting matrix. 
# <br/>This produces the equation for total memory (M):<br/>
# $M = 3(n^{2})+ 4$
# <br/><br/>
# We can then plot this equation for increasing values of <i>n</i>.

# In[67]:


#Plot increasing matrix size and memory for naive method
X_SizeN=[x for x in range(10,1000)]
Y_Bytes=[]
for x in X_SizeN:
    Bytes=3*4*(x**2)+4
    Y_Bytes.append(Bytes)

plt.scatter(X_SizeN,Y_Bytes,s=3,c='darkblue')
plt.xlabel("Square Matrix Dimension")
plt.ylabel("Bytes of Memory Taken to Execute")
plt.title("Memory requirements in Python relative to increasing dimensions of square matrix multiplication: Naive method")
plt.tight_layout()
plt.show()


# ### Matrix Multiplication: Strassen's Method

# Strassen developed an alternative matrix multiplcation method which follows a 'divide and conquer' strategy, dividing subject matrices into smaller ones and then building the resolutions of these back up to complete the original problem.<br/><br/>
# The method can only be conducted on square matrices where the dimensions (<i>n</i>) is power of 2. This method takes two matrices meeting this criteria, then divides them up into multiple 2 by 2 matrices, then performs matrix multiplication on each smaller matrix and sums resulting matrices. This is then recurrsively performed again on <i>n</i> size 2^2 until n dimensions are reached. 
# Since matrix addition (and subtraction) is performed element-wise, it is conducted in fewer steps than matrix multiplication. This method therefore reaches the same result in fewer steps than the traditional iterative method tested above.

# In[68]:


#Function to see if input is a power of 2
def IsPower2(A):
        m,n=len(A[0]),len(A[1])
        if not m == n:
            return False
        Power2=1
        while Power2 <= n:
            if Power2 == n:
                print("is Power 2")
                return True
            Power2 *= 2
        return False


#Function for Matrix Addition
def MatxAdd(A,B):
    n,m=len(A[0]),len(A[1])
    k,l=len(B[0]),len(B[1])
    if n!=k and m!=l:
        print("Matrices incompatible for matrix addition.")
        quit()
    else:
        NewMatrix=[[0 for col in range(n)] for row in range(m)]
        for i in range(n):
            for j in range(m):
                NewMatrix[i][j]=A[i][j]+B[i][j]
        return NewMatrix
    
#Function for Matrix Subtraction
def MatxSub(A,B):
    n,m=len(A[0]),len(A[1])
    k,l=len(B[0]),len(B[1])
    if n!=k and m!=l:
        print("Matrices incompatible for matrix subtraction.")
        quit()
    else:
        NewMatrix=[[0 for col in range(n)] for row in range(m)]
        for i in range(n):
            for j in range(m):
                NewMatrix[i][j]=A[i][j]-B[i][j]
        return NewMatrix

#Function to split a matrix into     
def SplitMatrix(A):
    n,m=len(A[0]),len(A[1])
    if n%2!=0 or m%2!=0:
        print("Unable to split matrix into two.")
        return np.nan
    elif n!=m:
        print("Matrix input not square.")
        quit()
    else:
        h=int(n/2)

        sub1=[A[i][:h] for i in range(h)]
        sub2=[A[i][h:] for i in range(h)]
        sub3=[A[i][:h] for i in range(h,h*2)]
        sub4=[A[i][:h] for i in range(h,h*2)]
        
        return(sub1,sub2,sub3,sub4)
    
#Function to compute strassen method
def StrassMatxMultp(A,B):
    n,m=len(A[0]),len(A[1])
    k,l=len(B[0]),len(B[1])
    
    if n<=2:
        return MatxMultipl(A,B)
    
    else:
        a,b,c,d=SplitMatrix(A)
        e,f,g,h=SplitMatrix(B)

        P1=StrassMatxMultp(a, MatxSub(g, h))
        P2=StrassMatxMultp(MatxAdd(a, b), h)
        P3=StrassMatxMultp(MatxAdd(c, d),e)
        P4=StrassMatxMultp(d, MatxAdd(f, e))
        P5=StrassMatxMultp(MatxAdd(a, d), MatxAdd(e, h))
        P6=StrassMatxMultp(MatxSub(b, d), MatxAdd(f, h))
        P7=StrassMatxMultp(MatxSub(a, c), MatxAdd(e, g))
    
        I = MatxSub(MatxAdd(MatxAdd(P5, P4), P6), P2)
        J = MatxAdd(P1, P2)
        K = MatxAdd(P3, P4)
        L = MatxAdd(MatxSub(MatxSub(P5, P3), P7), P1)
    
        FinalMatrix = [x[0]+x[1] for x in zip(I,J)] + [y[0]+y[1] for y in zip(K,L)]
    
    return FinalMatrix


# This method is timed on a small range of varying square matrices sizes. This is due to the constraints of only being able to compute matrices with dimensions to the power of 2, where increased sized matrices get much bigger very quickly.

# In[70]:


#Run the strassen function and time computation
#NB: this also takes quite a few minutes to run 
Dimensions2=[2**x for x in range(4,10,2)]
MeanRunTimes2=[]

for i in Dimensions2:
    Matrix1,Matrix2=np.random.randint(9,size=(i,i)),np.random.randint(9,size=(i,i))
    RunStats=TimeIt(StrassMatxMultp,Matrix1,Matrix2)
    MeanRunTimes2.append(RunStats[0])


# In[71]:


#Again we will plot the our x variable: matrix size against our y variable: mean run time (seconds)
plt.subplots(1,2, figsize=(7,3))
plt.suptitle("Average Run Time of Strassen's Matrix Multiplication against Matrix Size", fontsize=12)

#Plot raw data
plt.subplot(1,2,1)
plt.scatter(Dimensions2,MeanRunTimes2,s=20)
plt.xlabel("Dimension of Square Matrix")
plt.ylabel("Mean Time (Seconds)")

#Calculate and plot log10 of raw data
plt.subplot(1,2,2)
LogDimensions2=np.log10(Dimensions2)
LogMeanRunTimes2=np.log10(MeanRunTimes2)

plt.scatter(LogDimensions2,LogMeanRunTimes2,s=20,c='darkgrey')
plt.xlabel("log10 Dimension of Square Matrix")
plt.ylabel("Log10 Mean Time (Seconds)")
plt.tight_layout()
plt.show()


# We then make an approximation on runtime complexity using the gradient of this slope.

# In[72]:


slope,intercept,r_value,p_value,std_err = stats.linregress(LogDimensions2,LogMeanRunTimes2)
print("The gradient is:\n {} ".format(slope))


# From this, we can deduce that the order of run time for the strassen method of matrix multiplication is approximately $O(n^{2.7})$

# #### Memory Complexity: Strassen's Method

# 4 bytes of memory per integer or float is also applied to the calculation to estimate memory requirements for the Strassen's method of matrix multiplication with increasing sizes on square matrices.<br/>
# To calculate the memory requirments for the strassen method, we consider:<br/>
# <br/>1) 4 bytes of memory for every element in the both original matrices to be multiploed: $2 * n^{2}$
# <br/>2) 4 bytes of memory for the resultant matrix, also size <i>n</i> by <i>n</i>: $n^{2}$
# <br/>3) Then for each of the matrices to be multiplied, we take the log2 base of the number, then we recursively divide the matrix dimensions of the matrix by 2 up to unit 2, and for each of these units (m) we take 4 bytes times the m by m matrix multiplied by 4 and 2 resulting matrices also m by m size, multiplied by the 4 bytes.

# In[73]:


#Plot increasing matrix size and memory for strassen method
X_SizeN2=[x for x in range(10,1000)]
Y_Bytes2=[]
for x in X_SizeN:
    Bytes=3*4*(x**2)+4
    for i in range(2,x):
        if i%2==0:
            SubBytes=6*4*(i**2)
            Bytes+=SubBytes
    Y_Bytes2.append(Bytes)

plt.scatter(X_SizeN2,Y_Bytes2,s=3,c='blue')
plt.xlabel("Square Matrix Dimension")
plt.ylabel("Bytes of Memory Taken to Execute")
plt.title("Memory requirements in Python relative to increasing dimensions of square matrix multiplication: Strassen method")
plt.show()


# ### Iterative and Strassen Methods: Comparison

# As indicated by the 'O' notations of the first iterative method $O(n^{3})$ and the Strassen method $O(n^{2.7})$ indicates that the Strassen method is generally more efficient. This can visualise this by plotting both average run times on the same graph:

# In[74]:


MultScatter=plt.gca()
    
NM=MultScatter.scatter(Dimensions,MeanRunTimes, color="grey")
SM=MultScatter.scatter(Dimensions2,MeanRunTimes2, color="lightblue")
plt.legend((NM,SM),('Naive Method', 'Strassen Method'),scatterpoints=1, loc='upper left', ncol=2, fontsize=8)
plt.xlabel("Dimension of Square Matrix")
plt.ylabel("Mean Run Time (Seconds)")
plt.title("Mean Run Times of Naive and Strassen methods of Matrix Multiplication")
plt.tight_layout()
plt.show()


# Here we can see that for very small matrices, the mean run times for both methods are very similar and then they diverge.

# Memory of the two methods can also be compared by plotting both on the same graphs. 

# In[75]:


MemScatter=plt.gca()
    
N=MemScatter.scatter(X_SizeN,Y_Bytes, color="grey")
S=MemScatter.scatter(X_SizeN,Y_Bytes2, color="lightblue")
plt.legend((N,S),('Naive Method', 'Strassen Method'),scatterpoints=1, loc='upper left', ncol=2, fontsize=8)
plt.xlabel("Dimension of Square Matrix")
plt.ylabel("Mean Run Time (Seconds)")
plt.title("Memory of Naive and Strassen methods of Matrix Multiplication")
plt.tight_layout()
plt.show()


# The graph shows again for smaller sized matrices, the memory requirements for both approaches are similar however as matrix size increases, the Strassen method requires exponentially more memory than the naive approach.

# Although the naive method is favoured when focusing on memory, with the increases in memory storage and computing power in common devices, as well as opportunities to convert to a distributed computing model, the memory requirements are unlikely to cause much hinderance when implementing these methods on an individual or wider scale.<br/>
# The Strassen method is more limited in the requirements for inputs not only to be square matrices, but also those whose dimensions are a power of two. In common academic and commercial datasets bracketed under 'big data', used by data scientists where matrix multiplication may be needed, these criteria are unlikely to be met, thus making the Strassen method often unfeasible. There exists additional programming loops which can expand the Strassen method to make it more inclusive of other dimensions of matrices, but such additions add to the run time and increase the run order. There is also code complexity implications to consider, where excess and more complex code is often undersirable in collaborative working environments in terms of both construction and maintenance. 

# ### Conclusion

# The naive iterative approach and Strassen approach to matrix multiplication show two unique methods to compute the product of two matrices, and both have strikingly opposing advantages and limitations. Therefore, their implications will heavily depend on the conditions of the matrices to be multiplied.<br/> For large square matrices, whose dimensions are a power of two, the Strassens method is very favourable and the small reduction in the order of run time will save a lot of time for data scientists compared with the the naive method applied to the same matrices. Adaptations to the Strassen method, despite added coding complexity, may also be of further consideration, as even though they are likely to increase the order of run time, it will still compute much faster than the naive method fir very large matrices. When handling increasingly large matrices with the Strassen method, researchers and alike should also be aware of the exponential increased demands on computer memory required.<br/>
# For smaller matrices, particularly when the inputs are not square matrices, the 'naive' iterative approach is favourable. This is further advised when computation and coding skills of a team is limited, as the Strassen method additions which may accomodate for non-square and to the power of two matrices, may get complex and add on extra labour time in understanding the computation and code.

# ### References

# 1. Boschetti, A. and Massaron, L., 2015. Python data science essentials.

# In[ ]:




