Download Link: https://assignmentchef.com/product/solved-ml-homework1-bayesian-linear-regression
<br>
<h1>1           Bayesian Linear Regression</h1>

Given the training set <strong>x </strong>and corresponding label set <strong>t</strong>, we want to predict the label <em>t </em>of new test point <em>x</em>. In other words, we wish to evaluate the predictive distribution <em>p</em>(<em>t</em>|<em>x,</em><strong>x</strong><em>,</em><strong>t</strong>)<em>.</em>

A linear regression function can be expressed as below where the <em>φ</em>(<em>x</em>) is a basis function:

<em>y</em>(<em>x,</em><strong>w</strong>) = <strong>w</strong><sup>T</sup><em>φ</em>(<em>x</em>)

In order to make prediction of <em>t </em>for new test data <em>x </em>from the learned <strong>w</strong>, we will:

<ul>

 <li>Multiply the likelihood function of new data <em>p</em>(<em>t</em>|<em>x,</em><strong>w</strong>) and the posterior distribution of training set with label set.</li>

 <li>Take the integral over <strong>w </strong>to find the predictive distribution:</li>

</ul>

<em>.</em>

Now, please answer the following questions:

<ol>

 <li>Why we need the basis function <em>φ</em>(<em>x</em>) for linear regression? And what is the benefit for applying basis function over linear regression?</li>

 <li>Prove that the predictive distribution just mentioned is the same with the form</li>

</ol>

<em>p</em>(<em>t</em>|<em>x,</em><strong>x</strong><em>,</em><strong>t</strong>) = N(<em>t</em>|<em>m</em>(<em>x</em>)<em>,s</em><sup>2</sup>(<em>x</em>))

where

<em>s</em><sup>2</sup>(<em>x</em>) = <em>β</em><sup>−1 </sup>+ <em>φ</em>(<em>x</em>)<sup>T</sup><strong>S</strong><em>φ</em>(<em>x</em>)<em>.</em>

Here, the matrix <strong>S</strong><sup>−<strong>1 </strong></sup>is given by <strong>S</strong>

(hint: <em>p</em>(<strong>w</strong>|<strong>x</strong><em>,</em><strong>t</strong>) ∝ <em>p</em>(<strong>t</strong>|<strong>x</strong><em>,</em><strong>w</strong>)<em>p</em>(<strong>w</strong>) and you may use the formulas shown in page 93.)

<ol start="3">

 <li>Could we use linear regression function for classification? Why or why not? Explain it!</li>

</ol>

1

<h1>2           Linear Regression</h1>

In this homework, you need to predict the chance of being admit in base on relevant student resume data. The following two approaches need to be realized respectively:

<ul>

 <li><em>Maximum likelihood </em>approach (<em>ML</em>)</li>

 <li><em>Maximum a posteriori </em>approach (<em>MAP</em>)</li>

</ul>

model! Dataset provides total 500 students with 7 features. Can you use these features to predict the chance of admit for your own dream school?

One might consider the following steps to start the work:

<ol>

 <li>Download and check for the dataset.</li>

 <li>Create a new Colab or Jupyter notebook file.</li>

 <li>Divide the dataset into training and validation.</li>

</ol>

<strong>Dataset Description</strong>

<ul>

 <li>dataset X.csv contains 7 different resume feature served as the input.</li>

</ul>

GRE score, TOFEL score, University rating, SOP, LOR, CGPA, Research

<ul>

 <li>dataset T.csv contains the chance of admit regard as the target. Chance of Admit</li>

</ul>

<strong>Specification</strong>

<ul>

 <li>For those problems with <strong>Code Result </strong>at the end, you must show your result in your .ipynb file or you will get no</li>

 <li>For those problem with <strong>Explain </strong>at the end, you must have a clear explanation or you will get low points.</li>

 <li>You are also encouraged to have some discussion on those problem which is not marked as <strong>Explain</strong>.</li>

</ul>

<ol>

 <li>Feature select</li>

</ol>

In real-world applications, the dimension of data is usually more than one. In the training stage, please fit the data by applying a polynomial function of the form

<em>D                                  D          D</em>

<em>y</em>(<strong>x</strong><em>,</em><strong>w</strong>) = <em>w</em><sub>0 </sub>+ <sup>X</sup><em>w<sub>i</sub>x<sub>i </sub></em>+ <sup>XX</sup><em>w<sub>ij</sub>x<sub>i</sub>x<sub>j                               </sub></em>(<em>M </em>= 2)

<em>i</em>=1                                <em>i</em>=1 <em>j</em>=1

and minimizing the error function.

<ul>

 <li>In the feature selection stage, please apply polynomials of order <em>M </em>= 1 and <em>M </em>= 2 over the dimension <em>D </em>= 7 input data. Please evaluate the corresponding RMS error on the training set and valid set. (15%) <strong>Code Result</strong></li>

 <li>How will you analysis the weights of polynomial model <em>M </em>= 1 and select the most contributive feature? <strong>Code Result, Explain </strong>(10%)</li>

</ul>

<ol start="2">

 <li>Maximum likelihood approach

  <ul>

   <li>Which basis function will you use to further improve your regression model, Polynomial, Gaussian, Sigmoidal, or hybrid? <strong>Explain </strong>(5%)</li>

   <li>Introduce the basis function you just decided in (a) to linear regression model and analyze the result you get. (Hint: You might want to discuss about the phenomenon when model becomes too complex.) <strong>Code Result, Explain </strong>(10%)</li>

  </ul></li>

</ol>

<em>φ</em>(<em>x</em>) = [<em>φ</em><sub>1</sub>(<em>x</em>)<em>,φ</em><sub>2</sub>(<em>x</em>)<em>,…,φ<sub>N</sub></em>(<em>x</em>)<em>,φ<sub>bias</sub></em>(<em>x</em>)]

<ul>

 <li>Apply N-fold cross-validation in your training stage to select at least one hyperparameter(order, parameter number, …) for model and do some discussion(underfitting, overfitting). <strong>Code Result, Explain </strong>(10%)</li>

</ul>

<ol start="3">

 <li>Maximum a posterior approach

  <ul>

   <li>What is the key difference between maximum likelihood approach and maximum a posterior approach? <strong>Explain </strong>(5%)</li>

   <li>Use Maximum a posterior approach method to retest the model in <strong>2 </strong>you designed. You could choose Gaussian distribution as a prior. <strong>Code Result </strong>(10%)</li>

   <li>Compare the result between maximum likelihood approach and maximum a posterior approach. Is it consistent with your conclusion in (a)? <strong>Explain </strong>(5%)</li>

  </ul></li>

</ol>