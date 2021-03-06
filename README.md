# pcaScript
A simple script that performs principle component analysis on an n-dimensional feature vector for easy visualization.

To use the program simply provide a properly formatted csv file as a command line argument, or after starting the program.
The script parses the csv file, performs pca on the data, and generates figures corresponding to projections onto 
1, 2, and 3 dimensions.

The format for the csv file is as follows:

Rows represent a data point in n-dimensional space
  
The number of columns, n, is the dimension of the input data
  
The number of rows is the number of example data points being processed
  
All cells should contain number values, no text in the file
  
An example 10 dimensional data set is provided, data.csv, to test the program.

When run, the test data set creates the following outputs:

![data.csv Reduced to 2 Dimensions](https://github.com/devindewitt/pcaScript/blob/assets/2-D.png?raw=true)

Example data.csv input reduced to 2 dimensions.

![data.csv Reduced to 2 Dimensions](https://github.com/devindewitt/pcaScript/blob/assets/3-D.png?raw=true)

Example data.csv input reduced to 3 dimensions.


  
