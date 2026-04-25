# PISF

PISF is a prediction framework for stall flutter analysis. The full framework consists of two parts: an aerodynamic prediction module and a structural response prediction module.

## Overview

This repository currently provides the core implementation of the structural response prediction part in the PISF framework.  
The released code focuses on multi-step pitch angle prediction in stall flutter under varying structural parameters and flow conditions.

The structural response model integrates data-driven sequence learning with physics-based residual constraints to improve prediction accuracy and physical consistency.

## Current Contents

- Core code for the structural response prediction module
- Dataset construction and preprocessing
- Model training and inference
- Physics-residual-based evaluation

## Framework Description

The complete PISF framework is developed for stall flutter prediction and includes:

1. An aerodynamic prediction module for unsteady aerodynamic modeling
2. A physics-constrained structural response prediction module for pitch angle forecasting

At present, this repository only contains the structural response prediction part.

## Research Scope

The current implementation focuses on structural response prediction in stall flutter, with particular emphasis on multi-step pitch angle forecasting.  
It is intended for studying unsteady aeroelastic responses under different flow velocities and structural parameter settings.

## Notes

Additional codes, experiment scripts, and related modules are still being organized and uploaded.  
This repository will be continuously updated as the remaining components are prepared.

## Status

This project is under active development.  
More files and documentation will be added in future updates.
