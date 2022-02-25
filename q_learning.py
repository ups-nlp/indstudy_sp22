import os

runs = 50
steps = -1

for i in range(0, runs):
    os.system("python play.py " + str(steps) + " collector z-machine-games-master/jericho-game-suite/zork1.z5 -v 0")
    print("-> " + str(i + 1) + " completed....")