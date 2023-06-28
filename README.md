# Off-Policy Evaluation with Out-of-Sample Guarantees
This repository contains code to replicate the experimental results in:

Sofia Ek, Dave Zachariah, Fredrik D. Johansson, Petre Stoica. ["Off-Policy Evaluation with Out-of-Sample Guarantees".](https://openreview.net/pdf?id=XnYtGPgG9p) 2023.

## Abstract 
We consider the problem of evaluating the performance of a decision policy using past observational data. The outcome of 
a policy is measured in terms of a loss (aka. disutility or negative reward) and the main problem is making valid 
inferences about its out-of-sample loss when the past data was observed under a different and possibly unknown policy. 
Using a sample-splitting method, we show that it is possible to draw such inferences with finite-sample coverage 
guarantees about the entire loss distribution, rather than just its mean. Importantly, the method takes into account 
model misspecifications of the past policy -- including unmeasured confounding. The evaluation method can be used to 
certify the performance of a policy using observational data under a specified range of credible model assumptions.

## Results
The files named main_xxx replicates all the results in the paper.

### NHANES data
The National Health and Nutrition Examination Survey (NHANES) dataset used in some of the numerical experiments is available here:

Qingyuan Zhao, Dylan S Small, and Paul R Rosenbaum. ["Cross-screening in observational studies that test many hypotheses".](http://cran.nexr.com/web/packages/CrossScreening/index.html) 2018.

More details of the study can be found [here](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013).

### IHDP data
The Infant Health and Development Program (IHDP) dataset used in appendix is available here:

Christos Louizos, Uri Shalit, Joris Mooij, David Sontag, Richard Zemel, Max Welling 
["Causal Effect Inference with Deep Latent-Variable Models"](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP/csv), 2017.

For more details of the study see Jennifer L Hill ["Bayesian Nonparametric Modeling for Causal Inference"](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162).
