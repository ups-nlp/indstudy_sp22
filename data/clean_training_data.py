import re

def categorize_action(action:str) -> int:
    """
    Looks at the action and sees which module it came from

    @param a single action
    @return the module it came from
    """

    movements = {'north','south','east','west','up','down','northwest','southeast','northeast', 'southwest', 'go', 'climb'}

    action = action.split(' ')
    action = set(action)

    if bool(action & movements):
        return 0

    else:
        return 1

#Script

lines = []
readFile = open("dm_train_data.txt")
for line in readFile:
    line = re.sub('#','', line)
    list = line.split(',')
    action = list[1]
    moduleNumber = categorize_action(action)

    trainingLine = list[0] + "," + list[1] + "," + str(moduleNumber) + "\n"

    lines.append(trainingLine)
readFile.close()

trainingData = open("dm_train_data.txt", "w")
trainingData.writelines(lines)
