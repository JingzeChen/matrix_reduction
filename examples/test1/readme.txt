
for each file, the first line is the number of columns we are going to read


for each file, the second line is the offset range from the number of column.
E.g. 0 9 in the second line means there are 10 columns before the start of the following columns we are going to see.
This also means there are 10 lower simplices before the very first simplex in the file.
So, for 2D.txt, the first line is the indexes of the vertices 
For 3D.txt, the first line is the indexes of the edges 
For 4D.txt, the first line is the indexes of the triangles 


From third line onward, we see the actual matrix