# PINN Advection–Diffusion

**Physics-Informed Neural Network** implementation in Julia  
Solving the 2D advection–diffusion equation for ocean pollutant transport  
Built with [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl), [Lux.jl](https://github.com/FluxML/Lux.jl), and the SciML ecosystem


## Overview

This repository demonstrates:

1. **Finite Difference Reference**  
   - Generates a high-fidelity solution of  
     `u_t + v_x u_x + v_y u_y = D (u_{xx} + u_{yy})`
     
     on a 51×51×101 grid in ≈0.26 s.
     

2. **Data Noise Injection**  
   - Adds Gaussian noise (0.5 % of standard deviation) to mimic real sensors.

3. **PINN Setup**  
   - A mesh-free 9-layer × 128-neuron network (Tanh activations).  
   - Loss = PDE residual + IC loss (weight 500) + data-fit loss (weight 10).  
   - Mini-batch sizes: 200 collocation points + 1024 data/IC points.

4. **Training & Benchmark**  
   - Trained with ADAM (lr=0.006, 6000 iterations).  
   - Reports final loss, training time, and relative L₂ error. 
   - Plots: solution heatmaps (IC & final) and Adam loss history.

---

## Configuration Parameters

| Parameter                          | Default          | Description                                     |
| ---------------------------------- | ---------------- | ----------------------------------------------- |
| `T_final`                          | `0.25`           | Final simulation time                           |
| `X_max`, `Y_max`                   | `1.0`            | Spatial domain size                             |
| `d`                                | `0.01`           | Diffusion coefficient                           |
| `(vx, vy)`                         | `(0.5,0.5)`      | Constant advection velocities                   |
| `noise_level_*`                    | `0.01`,`0.005`   | Noise levels for BC, IC, and data               |
| `data_loss_weight`                 | `10.0`           | Weight for FDM data-fit loss                    |
| `ic_loss_weight`                   | `500.0`          | Weight for initial-condition loss               |
| `data_batch_size`, `ic_batch_size` | `1024`           | Mini-batch sizes for data & IC terms            |
| `Network`                      | `9×128`          | 9 hidden layers, 128 neurons each (Tanh)        |
| `Optimizer`                      | `Adam(lr=0.006)` | ADAM training, 6000 iters (then L-BFGS optional) |

