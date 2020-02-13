
# coding: utf-8

# # ENPM 673 Perception for Autonomous Robots

# # Assignment 1

# Question 1:Assume that you have a camera with a resolution of 5MP where the camera sensor is square shaped with a
# width of 14mm. It is also given that the focal length of the camera is 15mm.
# 1. Compute the Field of View of the camera in the horizontal and vertical direction. 
# 2. Assuming you are detecting a square shaped object with width 5cm, placed at a distance of 20 meters from
# the camera, compute the minimum number of pixels that the object will occupy in the image.

# Solution:  Answer in the report

# Question 2:
# Two files of 2D data points are provided in the form of CSV files (Dataset_1 and Dataset_2). The data
# represents measurements of a projectile with different noise levels and is shown in figure 1. Assuming that
# the projectile follows the equation of a parabola,
# 
# 1. Find the best method to fit a curve to the given data for each case. You have to plot the data and your
# best fit curve for each case. Submit your code along with the instructions to run it.
# 2. Briefly explain all the steps of your solution and discuss why your choice of outlier rejection technique is best
# for that case.

# First step:
# 
# We import the required libraries and read the dataset. We use the read_csv function from the Pandas library to access the csv file of the dataset.

# In[1]:
print("########################################################################################################")
print("                                          Solution 2:")
print("########################################################################################################")

#Import Dataset and read
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
data_1 = pd.read_csv(r"C:\Users\Kartik\Documents\ENPM673\Assignment 1\Dataset\data_1.csv")
data_2 = pd.read_csv(r"C:\Users\Kartik\Documents\ENPM673\Assignment 1\Dataset\data_2.csv")


# Next step:
# 
# We convert the dataset into a list and then split the data into separate x and y variable lists.

# In[2]:


data_1 = data_1.values.tolist()
data_2 = data_2.values.tolist()
#Split Data
x_data1 =[]
y_data1 =[]
x_data2 =[]
y_data2 =[]
for i in range(len(data_1)):
    x_data1.append(data_1[i][0])
    y_data1.append(data_1[i][1])
for i in range(len(data_2)):
    x_data2.append(data_2[i][0])
    y_data2.append(data_2[i][1])


# Next step:
# 
# We plot both the datasets using functions from the matplotlib library.

# In[3]:



plt.figure()
plt.plot(x_data1, y_data1,'o', label= 'dataset1')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Dataset 1")
plt.legend()

plt.figure()
plt.plot(x_data2, y_data2,'o', label = 'dataset2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Dataset 2")

plt.legend()
plt.show()


# Next step:
# 
# We define a function to compute the power of a list.

# In[4]:


def calc_power(list_val,power):
    out = []
    for i in range(len(list_val)):
        out.append(pow(list_val[i],power))    
    return out


# Next step:
# 
# We then define a function to determine the model parameters a, b, c of the quadratic equation so that we can predict the output y for any given input x.

# In[5]:



def build_Model(x,y):
    model_params=[]
    n= len(x)
    
    X = np.array([[       n           ,        sum(x)      ,sum(calc_power(x,2))],
                 [      sum(x)        ,sum(calc_power(x,2)),sum(calc_power(x,3))],
                 [sum(calc_power(x,2)),sum(calc_power(x,3)),sum(calc_power(x,4))]])
    xy  = [np.dot(x,y)]
    x2y = [np.dot(calc_power(x,2),y)]
    Y   = np.array([[(sum(y))],[(sum(xy))],[(sum(x2y))]])
    
    model_params = np.dot(np.linalg.inv(X),Y)

    return model_params


# Next step:
# 
# We define a function to predict the output y using the model parameters and the input data x.

# In[58]:


def predictOutput(x,y,A1):
    y_predict = A1[2]*calc_power(x,2)+A1[1]*x+A1[0]
    return y_predict



# Next step:
# 
# Plot the output curve fit using the LMSE method.

# In[59]:


plt.figure()
plt.plot(x_data1, y_data1,'o', label= ' Dataset 1')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Least Mean Square Output Curve Fit")

plt.plot(x_data1,predictOutput(x_data1,y_data1,build_Model(x_data1,y_data1)), color='red', label= 'Curve Fit')
plt.legend()
plt.show()

plt.figure()
plt.plot(x_data2, y_data2,'o', label= ' Dataset 2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Least Mean Square Output Curve Fit")
plt.plot(x_data2,predictOutput(x_data2,y_data2,build_Model(x_data2,y_data2)), color='red', label= 'Curve Fit')
plt.legend()

plt.show()


# Next step:
# 
# We then define a function that implements the RANSAC algorithm.
# 
# RANSAC algorithm:
#     1. Select three data points randomly from the dataset.
#     2. Determine the model parameters for the quadratic from those three data points.
#     3. Compare all the datapoints with the predicted model equation and classify them as inliers or outliers.
#     4. Select a model that maximizes the ratio of inliers to outliers.
#     5. Generate a curve fit from the final model.

# In[55]:


def ransac(x_data,y_data, n, t, success_threshold):
    final_inliers_x=[]
    final_inliers_y=[]
    final_outliers_x=[]
    final_outliers_y=[]
    worstfit= 0
    prev_inliers=0
    #Number of iterations
    # n=10000
    
    #Threshold value
    # t=55

    #Worst possible error is infinite error
    worst_error = np.inf                                 

    for i in range(n):
        
        dataPoints = random.sample(range(len(x_data)), 3)
        #print(dataPoints)
        possible_inliers_x=[]
        possible_inliers_y=[]
        
        for i in dataPoints:
            possible_inliers_x.append(x_data[i])
            possible_inliers_y.append(y_data[i])
        test_Model = build_Model(possible_inliers_x,possible_inliers_y)
        y_predict = predictOutput(x_data,y_data,test_Model)
        print(possible_inliers_x)
        print(possible_inliers_y)

        num_inliers =0
        num_outliers =0
        valid_inliers_x=[0]
        valid_inliers_y=[0]
        valid_outliers_x=[0]
        valid_outliers_y=[0]
        
        for i in range(len(x_data)):
            
            if abs(y_data[i]-y_predict[i]) < t:
                valid_inliers_x.append(x_data[i])
                valid_inliers_y.append(y_data[i])
                num_inliers+=1
            else:
                valid_outliers_x.append(x_data[i])
                valid_outliers_y.append(y_data[i])
                num_outliers+=1
        
        if num_inliers > worstfit:
            worstfit=num_inliers
            
            print("###############################    Better Model Found     #####################################")
            #Update chosen starting points
            
            input_points_x= possible_inliers_x
            input_points_y= possible_inliers_y
            
            #Update the model parameters
            update_model = build_Model(valid_inliers_x,valid_inliers_y)
            op= predictOutput(valid_inliers_x,valid_inliers_y,update_model)
            final_model = update_model
            
            #Update temperary variables to preserve data corresponding to the final chosen model.
            
            fin_inlier=num_inliers
            fin_outlier=num_outliers
            final_inliers_x=valid_inliers_x.copy()
            final_inliers_y=valid_inliers_y.copy()
            final_outliers_x=valid_outliers_x.copy()
            final_outliers_y=valid_outliers_y.copy()
            
            success_rate= (worstfit/len(x_data))*100
            
            if success_rate >= success_threshold: 
                break
            print(num_inliers,num_outliers)
        
    print(fin_inlier,fin_outlier)
     
    print('Worstfit=',worstfit)
    
    plt.figure()
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title("RANSAC Output Curve Fit")
    plt.plot(x_data,predictOutput(x_data,y_data,final_model), color='red',label='Curve Fit')
    plt.plot(x_data,y_data,'o', color='blue',label='Input Data')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title("RANSAC Output Curve Fit")
    plt.plot(x_data,predictOutput(x_data,y_data,final_model), color='red',label='Curve Fit')
    plt.plot(final_inliers_x,final_inliers_y,'o', color='black',label='Inliers')
    plt.plot(final_outliers_x,final_outliers_y,'o', color='orange',label='Outliers')
    plt.plot(input_points_x,input_points_y,'o', color='green',label='Picked Points')
    plt.legend()
    plt.show()


# Next step:
# 
# We plot the output curve fit for the first dataset using the RANSAC algorithm.

# In[56]:


ransac(x_data1,y_data1, 10000, 45,95)


# Next step:
# 
# We plot the output curve fit for the second dataset using the RANSAC algorithm.

# In[57]:


ransac(x_data2,y_data2, 10000, 45,95)


# Question 3:
# 
# The concept of homography in Computer Vision is used to understand, explain and study visual perspective,
# and, specifically, the difference in appearance of two plane objects viewed from different points of view. This
# concept will be taught in more detail in the coming lectures. For now, you just need to know that given
# 4 corresponding points on the two different planes, the homography between them is computed using the
# following system of equations Ax = 0, where:
# 
# 1. Show mathematically how you will compute the SVD for the matrix A. 
# 2. Write python code to compute the SVD. 

# Solution: 
# 
# 1. Solved in the Report
# 2. Code to compute SVD:
# 

# In[71]:

print("########################################################################################################")
print("                                          Solution 3:")
print("########################################################################################################")

import numpy as np
from numpy import linalg as LA
import pprint

#variable declaration and initialization

x1 =5
y1 = 5
xp1 = 100
yp1 = 100
x2 = 150
y2 = 5
xp2 = 200
yp2 = 80
x3 = 150
y3 = 150
xp3 = 220
yp3 = 80
x4 = 5
y4 = 150
xp4 = 100
yp4 = 200

# Matrix A
A = np.array([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],
             [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
             [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],
             [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
             [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3],
             [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
             [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],
             [0 , 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]],dtype='float64')

print("Matrix A is:\n",A)

# A transpose
At = np.transpose(A)
print("A transpose is given as:\n",At)

# A times A transpose
AAt = np.matmul(A,At)
print("A times A transpose is:\n",AAt)

# Eigen values and Eigen vectors of A times A transpose
eigenval_AAt, eigenvec_AAt = LA.eig(AAt)
print("Eigen value of A times A transpose is:\n",eigenval_AAt)

# A transpose times A
AtA = np.matmul(At,A)
print("A transpose times A is:\n",AtA)

#Eigen values and Eigen vectors of A transpose times A
eigenval_AtA, eigenvec_AtA = LA.eig(AtA)
print("Eigen value of A transpose times A is:\n",eigenval_AtA)

# the columns of U are the left singular vectors
U = eigenvec_AAt
print("The U matrix is:\n",U)

# V transpose has rows that are the right singular vectors
Vt = eigenvec_AtA
print("The V transpose matrix is:\n",Vt)

# S is a diagonal matrix containing singular values
S = np.diag(np.sqrt(eigenval_AAt))
S = np.concatenate((S,np.zeros((8,1))), axis = 1)
print("S matrix is given as:\n",S)

# The Homography matrix
H = Vt[:,8]
H = np.reshape(Vt[:,8],(3,3))
print("The homography matrix is:\n",H)

