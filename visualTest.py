import matplotlib.pyplot as plt
import numpy as np

file = open("data_file1.txt", "r")

arr = []

for line in file:
    #arr.append(int(line.split('(')[1].split(')')[0]))
    #arr.append(float(line.split(' ')[-2]))
    line_split = line.split('$')
    score = line_split[0]
    oops = line_split[1].split('||')
    next_obs = oops[0]
    action = oops[1]
    q_score = line_split[2]
    score = line_split[3]

    #arr.append([obs, act, float(score)])
    arr.append(float(score))
    

file.close()

diff_arr = np.array(arr)

plt.hist(diff_arr, bins = "auto")
print("---Overall Info---")
print("Data taken from " + str(len(arr)) + " states")
print("Min: " + str(np.min(diff_arr)))
print("Max: " + str(np.max(diff_arr)))
print("Mean: " + str(np.mean(diff_arr)))
#print("---Game Info---")
#print("Deaths: " + str(np.count_nonzero(diff_arr == -10)))
#print("Moves w/ no increase: " + str(np.count_nonzero(diff_arr == 0)))


# LOOOOOOK