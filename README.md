

<h1 align="center">Beyond The FRONTIER</h1>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/Tensorflow-1.13-green" alt="Vue2.0">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Author-JialiCui-blueviolet" alt="Author">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Email-cuijiali961224@gmail.com-blueviolet" alt="Author">
    </a>
    <a href="https://www.stevens.edu/">
        <img src="https://img.shields.io/badge/College-SIT-green" alt="Vue2.0">
    </a>
</p>


[Paper Here](https://arxiv.org/pdf/1710.10903.pdf)
## TODO
* somehow the loss and accuracy are not correct. Not sure where the bug is.
<br /><br />
## BackGround
* This is just a demo implementation of gat and the params are from [Author](https://github.com/tkipf/gcn)

<br />

## Quick Start
* Run train.py
* The hyper params are already set up in train.py.

<br />

## Version of Installment

#### Tensorflow 1.13.1

#### Numpy 1.18.2

#### Python 3.6.9  

<br />

### Dataset
#### In order to use your own data, you have to provide

* an N by N adjacency matrix (N is the number of nodes),

* an N by D feature matrix (D is the number of features per node),and

* an N by E binary label matrix (E is the number of classes). 

* Have a look at the loadData.py

Same as the author's code, we load citation network data Cora. The original datasets can be found here:[dataset](http://www.cs.umd.edu/~sen/lbc-proj/LBC.html.)

In our version (see data folder) we use dataset splits provided by [data spiliting](https://github.com/kimiyoung/planetoid)
(Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, Revisiting Semi-Supervised Learning with Graph Embeddings, ICML 2016).


<br />

## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
