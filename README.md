# location-optimization
This repo contains a small part of my work on using un-supervised learning to optimize the locations of antennas in a cellular communication network.

The goal is to use statistical data from locations of users to design the radio access network. This is only a small part of my work but conveys most of the ideas used in my research. 

The genral idea is to:
  1 - Use kernel density estimation to form a probability density function of the traffic
  2 - Apply either Lloyds quantization or k-harmonic means clustering algorithms to divide the users into regions and 
      put an antenna (base station or remote radio head) in each cluster


The files in this repo are as follows:

- placement_algorithms.py : A very simple script that performs a quick demo of the basic idea
- helpers.py : Helper functios for placement_algorithms.py
- ut_positions.mat : statistical data on locations of user terminals in the network (this is coming from a small simulated scenario, does not contain real world data but has almost the same format as the real data)
