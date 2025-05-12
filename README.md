# DADFS
***
**Data assimilation (DA) and degrees of freedom for signal (DFS) experiments**

This repository contains code for conducting data assimilation and DFS experiments as described in the submitted paper (Hu et al., 2025).

For any questions regarding the use of this code or experiments, please contact **guannan.hu@reading.ac.uk**.

---

## How to use
***
Run the `main_dadfs.py` file with different parameter settings to reproduce the experiments.

---

## Data assimilation algorithms
***
This project implements the following data assimilation algorithms:

- **Local Ensemble Transform Kalman Filter (LETKF)**  
  Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. *Physica D: Nonlinear Phenomena*, 230(1), 112–126. https://doi.org/10.1016/j.physd.2006.11.008

- **Ensemble Kalman Filter (EnKF)**  
  Evensen, G. (2003). The ensemble Kalman filter: Theoretical formulation and practical implementation. *Ocean Dynamics*, 53, 343–367. https://doi.org/10.1007/s10236-003-0036-9

---

## Approaches for estimating the DFS
***
Several approaches are implemented to estimate both the theoretical and actual Degrees of Freedom for Signal (DFS), as described in the submitted paper. These include both novel approaches proposed by the authors and established approaches from the literature.

In addition, the code includes a new strategy proposed by the authors for efficiently implementing DFS estimation approaches in the presence of domain localization.
