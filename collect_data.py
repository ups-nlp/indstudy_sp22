import os
from os.path import exists

runs = -1 # -1 for infinite
steps = -1 # -1 for infinite
data_dir = "data"
data_file_name_format = "data_file"
game_file_path = "z-machine-games-master/jericho-game-suite/zork1.z5"

#########

complete = 0
file_num = 0

while runs != 0:
    if not exists(data_dir):
        print(data_dir, "directory does not exist...")
        break

    cur_file_path = data_dir + '/' + data_file_name_format + str(file_num) + ".txt"
    if not exists(cur_file_path):
        open(cur_file_path, "a").close()
    else:
        while(exists(cur_file_path)):
            file_num += 1
            cur_file_path = data_dir + '/' + data_file_name_format + str(file_num) + ".txt"
        open(cur_file_path, "a").close()
    
    file_size = os.path.getsize(data_dir)
    file_size_gb = file_size / 1073741824.0
    print("File size at", file_size_gb, "gb")
    if file_size_gb > 10.0:
        break
        
    os.system("python play.py " + str(steps) + " collector " + game_file_path + " -v 0")
    complete += 1
    file_num += 1
    runs -= 1
    print("-> " + str(complete) + " completed....")