# indstudy_sp22
Independent Study Spring 2022

General Use: (I will update this more later i need to clean some stuff up)
    -The training and running process relies on there being a vocab.txt folder in a nearby GloVe-master
    -To train model run the command 'python play.py -1 qnet {gameFilepath} -v 0'
        -This will save the model weights periodically to a directory called nets in a file called qNet.npy
    -To test the model run the command 'python play.py {#ofMoves} testqnet {gameFilepath}
