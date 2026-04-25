# PISF

PISF is a stall flutter prediction framework consisting of two coupled modules: a large-model-fine-tuning-based aerodynamic prediction model and a physics-constrained structural response prediction model.

## Overview

This repository presents the core implementation of the PISF framework for stall flutter analysis under limited-data conditions. The framework is developed to improve both prediction accuracy and physical consistency in aeroelastic response prediction.

The overall framework contains two main modules:

### 1. Aerodynamic Prediction Module

This module is developed based on fine-tuning large models for time-series prediction. It is used to predict the instantaneous aerodynamic moment in stall flutter, which provides the aerodynamic input required for the physics-constrained structural response model.

### 2. Structural Response Prediction Module

This module is developed for structural response prediction in stall flutter, with a particular focus on multi-step pitch angle forecasting. During training, the predicted instantaneous aerodynamic moment from the aerodynamic module is introduced into the structural dynamic equation to construct physics-based constraints, thereby improving the physical consistency and prediction performance of structural response forecasting.

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
