# Welcome to eegGradients! 

eegGradients is a tool to help you make your own EEG gradients, as well as place your EEG time series into a gradient space. 
The toolbox is written in MATLAB using EEGLAB, and in Python using a numpy array as the input EEG time series data. You can use
MNE to integrate into this by using MNE's built-in `.get_data`.

## What are EEG gradients?

fMRI studies have coined gradients as [axes of brain organisation](https://www.sciencedirect.com/science/article/pii/S1053811922001161). 
The means by which gradients are derived in fMRI, are through 2 simple steps: 

1. Generate a functional connectivity matrix
2. Apply a dimensionality reduction technique to the generated matrix

EEG gradients translate these simple approaches used in fMRI, to EEG analysis. 
The main difference between application to fMRI data and EEG data is the connectivity matrix. 
EEG connectivity must consider the location of electrodes on the scalp in relation to other electrodes, 
since of course, electrodes measure potential differences.

## Repository Usage

Since application of EEG gradient analysis requires a few very simple techniques, this repository contains a small Python 
and MATLAB package that contains functions that carry out each of the steps for you.

## Package Installation

To get started in your chosen programming language, go to the sub-folder of this repository for your chosen programming 
language and follow the instructions there.

## Documentation

Documentation containing explanations of each function in the package (API), as well as tutorials with example
application scripts for both programming languages can be found [here]().
