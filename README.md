# Welcome to eegGradients! 

eegGradients is a tool to help you make your own EEG gradients, as well as place your EEG time series into a gradient space. Rather than a toolbox that requires a specific programming language, dependencies, and code, this repository is a guide for how to generate EEG gradients in any programming language, and will guide you through the process of doing so. It also takes the extra step of generating trajectories from your gradients. Don't worry, turns out it really isn't hard! :)

## What are EEG gradients?

fMRI studies have coined gradients as [axes of brain organisation](https://www.sciencedirect.com/science/article/pii/S1053811922001161). The means by which gradients are derived in fMRI, are through 2 simple steps: 

1. Generate a functional connectivity matrix
2. Apply a dimensionality reduction technique to the generated matrix

If you know what those things are and know how to apply those steps in the context of EEG, then that's all the info you need. Go off any apply those steps yourself! 
If you don't know what these things are, don't worry, we're going to go into it.

EEG gradients translate these simple approaches used in fMRI, to EEG analysis. There are some differences between EEG and fMRI application, which we will get into.

## Repository Structure

Each step in the process is indicated by an individual folder, which are numbered by their order. Go into each folder and the README.md file will show you how to carry out that step.

Each step has the mathematical run-down of how to carry it out. It also provides simple examples of code in multiple languages that you can follow along with, if you are using those code libraries. If you have a specific language that you would like to use that isn't included in examples, let me know and I can add it for you! 
