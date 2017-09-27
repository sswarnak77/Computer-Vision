
# coding: utf-8

# In[153]:


import numpy as np
import pandas as pd


# # A)

# In[154]:


A=[1,2,3]
B=[4,5,6]
C=[-1,1,3]


# In[155]:


A=np.array(A)
B=np.array(B)
C=np.array(C)


# In[156]:


print 2*A-B


# In[157]:


print "The Angle of A relative to positiove X-axis: ",1/np.sqrt(sum(pow(A,2)))


# In[158]:


print "Unit vector in the direction of A:",A/np.sqrt(sum(pow(A,2)))


# In[159]:


print "The directions of Cosines of A are:",1/np.sqrt(sum(pow(A,2))),",",2/np.sqrt(sum(pow(A,2))),"and",3/np.sqrt(sum(pow(A,2)))


# In[160]:


print "A.B:",np.dot(A,B)
print "B.A:",np.dot(B,A)


# In[163]:


print "The angle between A and B is:",np.dot(A,B)/np.sqrt(sum(pow(A,2)))*np.sqrt(sum(pow(B,2)))


# In[164]:


print "AxB:",np.cross(A,B)
print "AxB:",np.cross(B,A)


# In[165]:


print "transpose(A)*B:",np.matmul(np.transpose(A),B)
print "A*transpose(B):",np.matmul(A,np.transpose(B))


# In[166]:


print "A vector which is perpendicular to A and B:",np.cross(A,B)


# # B)

# In[167]:


A=[[1,2,3],[4,-2,3],[0,5,-1]]
B=[[1,2,1],[2,1,-4],[3,-2,1]]
C=[[1,2,3],[4,5,6],[-1,1,3]]


# In[168]:


A=np.array(A)
B=np.array(B)
C=np.array(C)


# In[169]:


print 2*A-B


# In[170]:


print "AB:",np.matmul(A,B)
print "BA:",np.matmul(B,A)


# In[171]:


print "transpose(AB):",np.transpose(np.matmul(A,B))


# In[172]:


print "transpose(B)*transpose(A):",np.matmul(np.transpose(B),np.transpose(A))


# In[173]:


print "|A|:",np.linalg.det(A)
print "|c|:",np.linalg.det(C)


# In[79]:


print "Inverse of A:",np.linalg.inv(A)


# In[175]:


print "Inverse of B:",np.linalg.inv(B)


# # C)

# In[176]:


A=[[1,2],[3,2]]
C=[[2,-2],[-2,5]]


# In[177]:


w,v=np.linalg.eig(A)


# In[178]:


print "EigenValues of A:",w


# In[179]:


print "EigenVectors of A:",v


# In[180]:


wb,vb=np.linalg.eig(B)


# In[181]:


print "inverse(V)AV:",np.linalg.inv(v)*np.matmul(A,v)


# In[182]:


print "Dot product between the eigenvectors of A:",np.dot(v[0],v[1])


# In[183]:


print "Dot product between the eigenvectors of B:",np.dot(vb[0],vb[1])


# In[184]:


print "The property of the eigenvectore of B:","Since B is symmetric real matrix so its eigenvectors are orthogonal"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




