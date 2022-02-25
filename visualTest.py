import matplotlib.pyplot as plt
import numpy as np

file = open("data_file.txt", "r")

arr = []

for line in file:
    #arr.append(int(line.split('(')[1].split(')')[0]))
    arr.append(float(line.split(' ')[-2]))

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