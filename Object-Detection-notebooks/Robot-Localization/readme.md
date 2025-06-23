### Robot World 1-D

First, imagine you have a robot living in a 1-D world. You can think of a 1D world as a one-lane road. 

<img src="Computer-Vision-Projects/Object-Detection-notebooks/images/road_1.png" width="50%" height="50%">

### Uniform Distribution
Since the robot does not know where it is at first, the probability of being in any space is the same
### Enviroment
1. The robot starts off knowing nothing; the robot is equally likely to be anywhere and so `p` is a uniform distribution.
2. Then the robot senses a grid color: red or green, and updates this distribution `p` according to the values of pHit and pMiss.
3. We normalize `p` such that its components sum to 1.
   
* The probability that it is sensing the color correctly is `pHit = 0.6`.
* The probability that it is sensing the wrong color is `pMiss = 0.2`

<img src='Object-Detection-notebooks/images/robot_sensing.png' width=50% height=50% />
we can incorporate **uncertain** motion into our motion update. We include the `sense` function that you've seen, which updates an initial distribution based on whether a robot senses a grid color: red or green. 

<img src='Object-Detection-notebooks/images/uncertain_motion.png' width=50% height=50% />
what happens to an initial probability distribution as a robot goes trough cycles of sensing then moving then sensing then moving, and so on? <br>
Recall that each time a robot senses it gains information about its environment, and everytime it moves,it loses some information due to motion uncertainty.
<img src='Object-Detection-notebooks/images/sense_move.png' width=50% height=50% />

