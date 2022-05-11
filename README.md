# Independent Study Spring 2022 - Module Based Approach

### Testing
To test the agent simply do the following:
1. Navigate inside the indstudy_sp22 folder
2. Run play.py or evaluate.py (which I would do inside the nlp_425 environment)

On my machine based on my file structure running evaluate.py looks like this:
  python evaluate.py 10 100 dep ../z-machine-games-master/jericho-game-suite/zork1.z5

A version of the neural network is already saved in the ./NN folder, so there is no need to run the keras_dm_nn.py or get_walkthroughs.py. 

---

### Corpus / Training Data
If you want to change the corpus you could do that by updating the get_walkthroughs.py file inside the data folder. The main thing in there you would have to change is the pathing to the z-machine-games folder to match your filesystem. If you wanted to then update the vectors and vocab, you would have to run GloVe on the walkthrough corpus yourself, then replace the files named vectors.txt and vocab.txt.

---

### Neural Netwwork
As for the neural network, it can definately be adjusted by making changes to the keras_dm_nn.py file. Just make sure when you're saving the model in the NN folder you delete the old model before creating a new one. Also inside the keras_dm_nn.py file there is the old code for the implementations of adaboost and the svm commented out at the bottom which you can take a look at.

---

### Patching
Currently the main thing I am determined to patch is that if a sentence is made up of entirely novel words (ie. words not contained in the walkthrough) it could throw a divide-by-zero error. This is something I have only ran into on incredibly rare edge cases, so shouldn't currently cause any issues. I am planning on fixing this by using the <unk> vector when we come across an unknown word in an observation to remove that possibility.
 
---
  
### Acknowledgments
Special thanks to Prof. Chambers, Dannielle Dolan, and Penny Rowe for their awesome contributions!
