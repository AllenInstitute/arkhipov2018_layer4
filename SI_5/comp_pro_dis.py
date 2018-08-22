import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.genfromtxt('../build/ll2.csv', delimiter=' ')
index = data[1:8501,0]
location = data[1:8501,2:4] #2:5]

#find the first value that is larger than the 8500 threshold
def find_value(tar):
    for index in range(0,len(tar)):
        if(tar[index]>8500):
            return index

#read the orientation data
orien = np.genfromtxt('Ori/ll2_pref_stat.csv', delimiter=' ')
# A
#orien = orien[1:8501,1]
orien = data[1:8501,5]

# make a connectivity matrix
f_list = []
C = np.zeros([8500,8500])

for x in range(0,85):
    if(x == 0):
        f_list.append('../build/ll2_connections/target_0_%d00.dat'%(x+1))
        continue
    f_list.append('../build/ll2_connections/target_%d00_%d00.dat'%(x,x+1))

for f_name in f_list:
    print 'Building connectivity matrix using file %s.' % (f_name)
    tar = np.genfromtxt(f_name,delimiter=' ')
    info = tar[:,1]
    limit = find_value(info)

    con_x = tar[0:limit,0]
    con_y = tar[0:limit,1]
    for index in range(0,len(con_x)):
        x , y = con_x[index], con_y[index]
        C[x, y] = 1

def normalize(delta_theta):
    delta_theta = np.mod(delta_theta, 180)
    delta_theta[delta_theta > 90] -= 90
    return delta_theta

#Calculate the Euclidean distance
from scipy.spatial import distance
i_arr = np.array([])
j_arr = np.array([])

for x in range(0, 8500):
    if (x % 200 == 0):
        print 'Computing distances; progress %f%%.' % ((100.0*x)/8500)
    for y in range(x + 1, 8500):
        dis = distance.euclidean(location[x, :], location[y, :])
        if(dis < 50):
            i_arr = np.append(i_arr,[x])
            j_arr = np.append(j_arr,[y])

#calculation the difference in orientation
i = i_arr.astype(int)
j = j_arr.astype(int)

delta_theta = orien[i] - orien[j]
delta_theta = normalize(delta_theta)

num_pair, bins, patches = plt.hist(delta_theta, bins = [0,30,60,90])

i_conn = np.array([])
j_conn = np.array([])

for index in range(0, len(i_arr)):
    x , y = i_arr[index], j_arr[index]
    if(C[x, y] == 1):
        i_conn = np.append(i_conn,[x])
        j_conn = np.append(j_conn,[y])

i_co = i_conn.astype(int)
j_co = j_conn.astype(int)

delta_conn_theta = orien[i_co] - orien[j_co]
delta_conn_theta = normalize(delta_conn_theta)

num_conn_pair, bins, patches = plt.hist(delta_conn_theta, bins = [0,30,60,90])

# Calculate the probability distribution
p = num_conn_pair/num_pair

plt.bar([0, 45, 90], p)
plt.ylim(0, 0.5)
print p

print orien[i][500:510]
print orien[j][500:510]
print delta_theta[500:510]


