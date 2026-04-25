# PISF

PISF is a prediction framework for stall flutter analysis, consisting of two main modules: an aerodynamic prediction model based on fine-tuning large language models, and a structural response prediction model based on physics-constrained learning.

## Overview

This repository provides the core implementation of the PISF framework for stall flutter prediction. The framework is designed to improve both prediction accuracy and physical consistency under limited-data conditions.

The current framework contains two main modules:

1. **Aerodynamic Prediction Module**  
   A large-model fine-tuning-based aerodynamic prediction model is developed to forecast unsteady aerodynamic quantities in stall flutter, providing the aerodynamic input for subsequent structural response prediction.

2. **Structural Response Prediction Module**  
   A physics-constrained structural response prediction model is constructed to perform multi-step pitch angle forecasting under varying structural parameters and flow conditions.

## Current Contents

- Core code for the aerodynamic prediction module
- Core code for the structural response prediction module
- Dataset construction and preprocessing
- Model training and inference
- Physics-residual-based evaluation

## Research Scope

The current implementation focuses on stall flutter prediction, with particular emphasis on aerodynamic forecasting and pitch angle response prediction.  
It is intended for studying unsteady aeroelastic behavior under different flow velocities and structural parameter settings.

## Notes

Additional codes, experiment scripts, and related modules are still being organized and uploaded.  
This repository will be continuously updated as the remaining components are prepared.

## Status

This project is under active development.  
More files and documentation will be added in future updates.
