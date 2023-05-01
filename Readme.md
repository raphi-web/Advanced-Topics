# Advanced Topics of GI-Science Source Code
Contains the implementations for Assignment 1

## Google Colab Link: 
https://drive.google.com/file/d/10VYR8PrnFDGCtkWtHhuhXrduRr9Z07vW/view?usp=sharing
## Source Code of Implementations:
https://github.com/raphi-web/Advanced-Topics

## Moving Points Voronoi Diagram
Can be viewd here: https://kestrel.page/static/projects/voronoi/www/index.html  
Source Code: https://github.com/raphi-web/voronoi

## Bonus because in the process of making the Voronoi Diagramm I made also a Delauney Version
Can be viewed here https://kestrel.page/static/projects/delauney/www/index.html  
Source Code: https://github.com/raphi-web/delauney-wasm

1. Generate a set of random points in 2D and find the closest pair of points. Is your program
efficient (in terms of execution time)? Can you improve it – how? (10)  



2. Given two random sets of 2D points, S1 and S2, write a function that finds the k-nearest
neighbors in S2 for each point in S1. (10)



3. Finding the k-nearest neighbors can have many practical applications. In the transportation
domain, for example, you can use the function to find the nearest public transport stop
(Haltestelle) from your apartment.
a) To accomplish the task, find the nearest stop/Haltestelle for all buildings in Graz city. Two
point datasets you can use; (30)



• Open street map (OSM) for getting buildings (centroid/location of buildings) data. You
can use different tools for extracting Graz building data from OSM e.g. OSMnx
• Graz Haltestellen/stops data can be downloaded from here
b) Perform analysis using visualizations (try to be innovative) (20)
4. Perform overlay operation (using intersection) on sets of geometries (e.g. two or more sets of
polygons). Check the efficiency of your program with respect to the number of input
geometries. (10)



5. Given a point and set of 2D polygons and 3D polyhedra, perform point in polygon/polyhedra
(PIP) test. (20)

Optional
Create an animation of Voronoi diagrams for moving points (20)
Think of an application for moving point Voronoi diagrams. Describe the scenario and what
problem/question the moving points Voronoi should answer in a few sentences. Either simulate
some moving point data or, if possible, find some real-life data to use. Create an animation
showing the results.

Network analysis in Python using Networkx (30)
1. Download the routable road network of Austria from OSM using OSMnx
2. Find shortest path between two given points using shortest path function of Networkx
3. Perform network statistics
4. Perform street network orientation for some cities of Austria
https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0189-1