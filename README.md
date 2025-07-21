# SPECTRA: Spectral Perceptual Experimentation and Cross-Testing Research Adversary

Haters will say that I just wanted a dope project name and then tried to find some way to justify it with an acronym afterwards. I will neither confirm nor deny these baseless accusations.

## Disclaimer
Spectra is inteded for research and educational use only. Users are responsible for ensuring ethical usage. The decision to open-source the system was largely made for the following reasons:

1. Gradient-based attacks tend to be algorithmically trivial. A motivated attacker could very easily create one. The real technical difficulty involved with this project so far has largely been due to architectural complexity, which would not provide too many advantages from an attacker's perspective, and largely exists to support more effective experimentation and testing. 

2. It could be argued that this system could be used to develop more advanced gradient-based evasion attacks. This is valid, but developing a new attack is much more difficult than simply implementing one. If an attacker was technically competent enough to develop such an attack, it's highly unlikely that they would be dependent on a tool like this to do so.

3. I believe that there is value in providing a reasonably well-written adversarial test harness in an open-source context. 

4. All the hash algorithms that this system currently supports are toy algorithms and should not be used in real-word contexts regardless, save for PDQ. PDQ was developed by Meta, and it's highly likely that they have already internally tested their pipelines using it against attacks more powerful than what this system is capable of producing. 

This repo implements Meta's PDQ algorithm and contains some original PDQ code too, which is licensed under the BSD license. You can find the original implementation [here](https://github.com/facebook/ThreatExchange/tree/main/pdq).

## Overview
Spectra is a comprehensive adversarial evaluation toolkit for perceptual hash algorithms. Inspired by [Foolbox](https://github.com/bethgelab/foolbox), Spectra makes it easy to benchmark the robustness of perceptual hash pipelines against black and white-box evasion based attacks. It's designed to be easily extensible in order to accomodate new attack strategies and hash algorithms, and also supports a very flexible experiment orchestration pipeline that can be used for quick one-off tests or highly comprehensive experiments alike.


## How To Use

You'll probably want to first clone this repo and then look at the code. After you've done this, try running an example experiment through the `run_attacks.py` script like so:

```bash
git clone https://github.com/nolanmccleary/Spectra.git
python run_attacks.py -f pdq_attackmode_comparison.yaml
```

Or any other experiment you desire. You can also write your own, which is the primary purpose of this project. Please note that depending on your hardware, some experiments may take a while to run. 

## Runtime Configuration

You can look inside spectra/config/attack_config.py to view the configuration scheme for experiments and attacks. Very briefly, Spectra is designed to primarily execute at the experiment level. Experiments contain one or more attacks, and also specify the target files used by all attacks that make up the experiment. All other attack configuration info is stored within the attack configuration, and `run_attacks.py` provides overrides for them as well. Please note that in its current state, these overrides are applied throughout the entire scope of the experiment, so explicitly setting the individual experiment configurations is recommended when running larger tests. 

Experiment and attack configurations can be set in three separate ways: 
1. From a YAML file 
2. From a dictionary
3. Directly through the constructor

For most cases, it's generally recommended to use either approach 1 or 3.

The fields for experiment and attack configuration are resolved at the time of experiment execution, meaning that initializing configs with incomplete field data is permitted provided that all fields have been filled appropriately when the experiment is actually run. Some fields also have defaults (i.e. default verbosity is low), and will resolve to them if the field is not explicitly configured or overriden. 


## Delta Generation

Spectra is designed to take an input image tensor and create an optimal perturbation delta based on the current attack configuration. That delta is then post-processed according to the attack configuration's settings and added to a copy of the original image to create the adversarial image tensor. The process of generating that delta is performed by an Optimizer instance. Multiple instances are already present in order to facilitate a few different attack strategies (mainly for the sake of comparison) but the stanadardized interface makes it easy to add more depending on what sort of behaviour is desired. Strictly speaking, as long as there is some sort of loss function to define the optimization criteria, the `NES_Optimizer` should work relatively well in non-differentiable contexts. It's clearly overkill in a differentiable situation, but at that point just use Autograd lol. 






## Contributions

If you have a contribution that you think would make the system better, please open up a pull request. Some things that would be good to implement at some point:

1. More loss function options to support different attack objectives beyond simple gradient ascent. For example, a Carlini Wagner optimization mode. 
2. A better pipeline for user-specified hash, loss, and acceptance functions that doesn't require adding to the existing source code (i.e. being able to take custom function hooks inside the config instead of a string and mapping it to a function, which is kind of dumb).
3. Smarter override logic in run_attacks.poy such that you can bind overrides at the level of an individual attack or experiment. Right now they are applied across all experiments, which is kind of dumb.
4. Add an option to generate an attack artifact that stores all calculated delta tensors. Useful for training some sort of attack network in the future. This would be a really cool project and nobody has done it yet afaik.


I might get around to implementing at least the first three of these myself, but at the moment I'm kinda sick of working on this thing, and I have a way more exciting project on the radar that I really wanna get started with. 
