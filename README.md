# Causal humans

The project is about analyzing the data obtained by conducting survey of 33.000 workers on work life. Firstly, the data is preprocessed into numerical data, and the artifacts are removed. Secondly, the dimensionality of the data is reduced  in order to be able to apply algorithms. Thirdly, clustering is made with the *k-means* algorithms, to create clusters. Fourthly, after studying the clusters the data is analyzed from a causal point of view.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisities

The project is made in Python 2.7 and is using mutiple libraries, which are : 

* Numpy
* Scipy
* Matplotlib
* Scikit-Learn

To install them, please run the lines below (Linux) : 

```
$ sudo apt-get install python-numpy
$ sudo apt-get install python-scipy
$ sudo pip install matplotlib
$ sudo pip install -U scikit-learn
```

This project requires no installation, just running the different *.py* files. Please note that many of the files were made specifically for the used dataset, which I'm not allowed to upload. The consequences are that most of the tests aren't runnable without modifiying some parameters in the code. The inputs and outputs are often *.csv* files, in order to read them easily with a office suite. However, the panda library isn't used to read those files : it is either the more simple *numpy.loadtxt()* or the archaic *csv.reader/writer()*, to process the lines one by one.

## How-to run the project files

** Soon **

## Authors

* **Diviyan Kalainathan** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


