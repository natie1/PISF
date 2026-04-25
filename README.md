# PISF

PISF is a prediction framework for stall flutter, consisting of two main modules: a large-model fine-tuning-based aerodynamic prediction model and a physics-constrained structural response prediction model.

## Overview

This repository provides the core implementation of the PISF framework for stall flutter prediction under limited-data conditions. The framework is developed to improve both prediction accuracy and physical consistency in aeroelastic response forecasting.

The current framework contains two major modules:

### 1. Aerodynamic Prediction Module

This module is designed to predict unsteady aerodynamic quantities in stall flutter.  
It takes historical pitch-angle sequences and flow velocity as inputs, and outputs the aerodynamic moment coefficient \(C_m\).

The current implementation is based on fine-tuning a large-model backbone for time-series prediction.  
A Qwen-based model is used as the backbone, combined with sequence patching, temporal embedding, and convolution-enhanced feature extraction.

### 2. Structural Response Prediction Module

This module is developed for structural response prediction in stall flutter, with a particular focus on multi-step pitch-angle forecasting.  
It integrates data-driven sequence modeling with physics-based residual constraints to improve physical consistency and prediction stability under varying structural parameters and flow conditions.

## Current Contents

- Core code for the aerodynamic prediction module
- Core code for the structural response prediction module
- Dataset construction and preprocessing
- Model training and inference
- Physics-residual-based evaluation

## Research Scope

The current implementation focuses on stall flutter prediction for aeroelastic systems.  
The aerodynamic module is used to predict unsteady aerodynamic moment coefficients, while the structural module is used to forecast pitch-angle responses under different flow velocities and structural parameter settings.

## Notes

Additional codes, experiment scripts, and related modules are still being organized and uploaded.  
This repository will be continuously updated as the remaining components are prepared.

## Status

This project is under active development.  
More files and documentation will be added in future updates.
