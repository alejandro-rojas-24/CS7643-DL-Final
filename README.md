# CS7643-DL-Final
Repository to host our code and output for our final project

Based off of the paper [SimLM: Can Language Models Infer Parameters of Physical Systems?](https://arxiv.org/abs/2312.14215)

## Project Overview
This project explores the capabilities of language models in understanding and inferring parameters of physical systems. We implement physics simulations using Pymunk to generate training data and test the ability of language models to predict physical parameters from system descriptions.


## Project Structure
- `physics_sim/`: Contains physics simulation code
  - `simple_projectile_example.py`: Example implementation of a projectile motion simulation

## Usage
The project currently includes a simple projectile motion simulation that demonstrates:
- Physics-based simulation using Pymunk
- Ground geometry generation
- Projectile trajectory calculation
- Visualization of results

To run the example simulation:
```bash
python physics_sim/simple_projectile_example.py
```
