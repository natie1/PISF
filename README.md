# PISF

PISF is a prediction framework for stall flutter analysis, consisting of two coupled modules: a large-model-fine-tuning-based aerodynamic prediction module and a physics-constrained structural response prediction module.

## Overview

This repository presents the core implementation of the PISF framework for stall flutter prediction under limited-data conditions. The framework is developed to improve both predictive accuracy and physical consistency in aeroelastic response analysis.

The overall framework contains two main modules:

### 1. Aerodynamic Prediction Module

This module is developed based on fine-tuning large models for time-series prediction.  
It takes historical pitch-angle sequences and flow velocity as inputs, and outputs the instantaneous aerodynamic moment coefficient.

The predicted aerodynamic moment is further used in the structural response module to construct physics-based constraints through the structural dynamic equation.

### 2. Structural Response Prediction Module

This module is developed for structural response prediction in stall flutter.  
It takes historical pitch-angle sequences, flow velocity, and structural parameters \((K, C, I)\) as inputs, and outputs future multi-step structural responses.

By integrating data-driven sequence modeling with physics-based residual constraints, this module is designed to improve both prediction accuracy and physical consistency under varying structural parameters and flow conditions.

## Current Contents

- Core code for the physics-constrained structural response prediction module
- Dataset construction and preprocessing
- Model training and inference
- Physics-residual-based evaluation

## Research Scope

The current framework focuses on stall flutter prediction for aeroelastic systems.  
In particular, the aerodynamic module is used to estimate the instantaneous aerodynamic moment coefficient, while the structural response module is used to predict future pitch-angle responses under different flow velocities and structural parameter settings.

## Notes

Additional codes, experiment scripts, and related modules are still being organized and uploaded.  
This repository will be continuously updated as the remaining components are prepared.

## Status

This project is under active development.  
More files and documentation will be added in future updates.
