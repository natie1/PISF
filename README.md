# PISF

PISF is a prediction framework for stall flutter analysis. It consists of two main modules: a large-model-fine-tuning-based aerodynamic prediction module and a physics-constrained structural response prediction module.

## Overview

This repository presents the core implementation of the PISF framework for stall flutter prediction under limited-data conditions. The framework is developed to improve both predictive accuracy and physical consistency in aeroelastic response analysis.

The overall framework contains two main modules:

### 1. Aerodynamic Prediction Module

This module is designed for unsteady aerodynamic prediction in stall flutter. It is developed based on fine-tuning large models for time-series prediction and is used to estimate aerodynamic quantities required for subsequent response analysis.

### 2. Structural Response Prediction Module

This module is developed for structural response prediction in stall flutter, with a particular focus on multi-step pitch angle forecasting. It integrates data-driven sequence modeling with physics-based residual constraints to improve prediction accuracy and physical consistency under varying structural parameters and flow conditions.

## Current Contents

- Core code for the physics-constrained structural response prediction module
- Dataset construction and preprocessing
- Model training and inference
- Physics-residual-based evaluation

## Research Scope

The current framework focuses on stall flutter prediction for aeroelastic systems. In particular, the present implementation emphasizes structural response forecasting, especially pitch angle prediction under different flow velocities and structural parameter settings.

## Notes

Additional codes, experiment scripts, and related modules are still being organized and uploaded.  
This repository will be continuously updated as the remaining components are prepared.

## Status

This project is under active development.  
More files and documentation will be added in future updates.
