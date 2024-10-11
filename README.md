# 26 week Machine Learning and Artificial Intelligence course

Hello and welcome to my 26 week ML and AI course. This course is designed for year 12 students (or 11th grade US students), who have some familiarity with calculus/pre-calculus and basic coding principles. 

The goal of this course is to start from the fundamentals of linear algebra, probability theory and multivariate calculus and eventually build up to implementing and training your own neural networks by the end of the course. The course finishes with a 3-4 week capstone project where you can sort of do whatever you like, but you are perfectly free to spend longer.  

Materials will be released each week from my private repository for you to have a go at, this includes both lecture notes/slides and some programming exercises in Python. Don't worry if you fall a bit behind we probably won't rigorously stick to the schedule and can spend more time on trickier things. 

## Video materials for linear algebra

[3Blue1Brown essence of linear algebra](https://youtu.be/kjBOesZCoqc?si=UUkI12_ND45JzhCQ)

## Course overview

Week 1: Introduction to the course, setup and basic Python programming

Week 2: Fundamentals of linear algebra, vector spaces, multiplication, transposition, linear transformations, matrix inverse, determinant, eigenvalues, eigenvectors.

Week 3: Basics of multivariate calculus, funtions of multiple variables, partial derivatives, vectorized gradients, critical points, Hessian matrix, Jacobians, simple linear regression, least squares estimate.

Week 4-5: Linear regression in depth, multiple linear regression, mathematical derivation of the least squares estimate, multivariate case, regularization with lasso and ridge regression, polynomial basis expansion.

Week 5-6: Fundamentals of probability theory, events outcomes, random variables, basic probability distributions, conditional probability, Bayes' Theorem, independence, expectation, variance, covariance, correlation, discrete and continuous variables.

Week 8: Introduction to data preprocessing and feature engineering, hands on with pandas and numpy.

Week 9: Introduction to machine learning, supervised v.s. unsupervised learning, training and test set split, simple linear regression and k-nearest neighbours.

Week 10-11: TBD

Week 11-12: Introduction to classification, logistic regression, naive bayes, performance metrics (precision, recall F-1 score).

Week 13-14: Hands on unsupervised learning, K-means, heirarchical clustering.

Week 15: Introduction to nerual networks, neurons, activation functions, feed forward neural networks, backpropagation.

Week 16: In depth analysis of neural networks, derivation of backpropagation for feed forward neural networks.

Week 17-18: Training neural networks, setup pytorch, desiging neural network architecture, loss functions and optimization algorithms (SGD, SGD with momentum, Adam), understanding learning rates and epochs.

Week 19: Advanced topics with neural networks, regularization techniques, dropout, batch norm, weight decay, model evaluation and overfitting.

Week 20: Introduction to convolutional neural networks, convolutions, pooling layers, hands on implementation details.

Week 21: Introduction to recurrent neural networks, time series forcasting and sequential data, vanishing gradient problem, LSTM and GRU.

Week 22: Introduction to reinforcement learning, agents, states, actions, rewards, Q learning and deep Q learning, applications.

Week 23-26: Capstone project based on any of the main topics introduced throughout the course.

## Installation and setup instructions

Please install conda via [Anaconda](https://www.anaconda.com/download/success) and make sure you install the correct distribution for your machine (Mac/Windows/Linux).

Now execute the following instructions in your terminal (Mac), cmd (Windows), or Debian/Ubuntu (Linux)

```conda info --envs```

Running this command you should see one environment listed ```base```. We are going to create a clone of the base environment called ```ml``` which should have everything we need installed.

```conda create --name ml --clone base```

Now run ```conda info --envs``` again and you should see the environment ```ml``` listed. When we are coding we always want to have our ```ml``` virtual environment active rather than our ```base``` environment because we are going to install things to the ```ml``` environment and keep them separate. To activate the ```ml``` environment run the following:

```conda activate ml```

You should see the ```ml``` environmengt is now active. You can run the python command line interpreter by executing the following line:

```python```

Now the python command line interpreter is active try running a few simple commands like ```1+1```, ```1>2``` and ```100/5```. To exit the python interpreter execute ```exit()```. We will do the majority of our coding in jupyter notebooks but sometimes its helpful to run a few things in the command line when you are debugging your code. First let's create a new working directory:

```mkdir new_dir```

Then let's navigate to this new directory:

```cd new_dir```

Let's list everything in the new directory (Mac/Linux):

```ls```

Or for Windows:

```dir```

The directory should be empty because we just created it. Now let's open jupyter notebook in this directory simply run:

```jupyter notebook```

Your brower should then open and you can create a new notebook and start coding in python!

## Setting up PyTorch

We won't need this for the first half of the course but you are welcome to set this up in advance or when we start using neural networks. Open your terminal/cmd/Debian/Ubuntu and activate your conda environment:

```conda activate ml```

Now go to the [PyTorch](https://pytorch.org/get-started/locally/) website and execute the corresponding command in your terminal. The command may change whether you have a Mac, Windows or Linux. And whether you have a CUDA enabled GPU or not. Some examples, 

Mac or Windows (without GPU):

```pip3 install torch torchvision torchaudio```

or 

```pip install torch torchvision torchaudio```

Linux (without GPU):

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```

Windows with GPU and latest CUDA driver (12.4+):

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124```

or

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124```

If you want to use PyTorch with CUDA enabled make sure your GPU driver is up-to-date and you know what the maximum compatibility is. Most of the time you can work this out by running:

```nvidia-smi```

And check the CUDA version.




