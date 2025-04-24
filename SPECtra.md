# SPECtra


## Architecture

1. Engine: Spectra_Engine - orchestrator + attack engines (one for each attack)

2. TestBench: Spectra_Testbench - Loads engine, stages attacks, consolidates attack results


## Roadmap

1. Get best engine possible

2. Define testbench architecture







## 04/23/2025


First ups:

- Put Gradient engine directly into attack object. Monolithic attack objects for ervery attack algo/tensor config will be easier to deal with

- Get Spectra engine handler able to work on individual files, folders, and a json attack specifier (most important)

- Figure out best way to plot attack curves

- Torched DCT util

- Get GPU compatibility up for MPS

- Get GPU compatibility up for CUDA

- Get PDQ + other hashes tested

- Build image quantization check into attack algo