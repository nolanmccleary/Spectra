# SPECTRA: Spectral Perceptual Experimentation and Cross-Testing Research Adversary

Haters will say that I just wanted a dope project name that sounded tangentially related to the project context and then tried to find some way to justify it with an acronym afterwards. I will neither confirm nor deny these baseless accusations.

## Disclaimer
Spectra is intended for research and educational use only. Users are responsible for ensuring ethical usage. The decision to open-source the system was made for the following reasons:

1. Gradient-based attacks tend to be algorithmically trivial. A motivated attacker could very easily create one. The real technical difficulty involved with this project so far has largely been due to architectural complexity, which would not provide too many advantages from an attacker's perspective, and largely exists to support more effective experimentation and testing. 

2. It could be argued that this system could be used to develop more advanced gradient-based evasion attacks. This is valid, but developing a new attack is much more difficult than simply implementing one. If an attacker was technically competent enough to develop such an attack, it's highly unlikely that they would be dependent on a tool like this to do so.

3. All the hash algorithms that this system currently supports are toy algorithms and should not be used in real-world contexts regardless, save for PDQ. PDQ was developed by Meta, and it's highly likely that they have already internally tested their pipelines using it against attacks more powerful than what this system is capable of producing. That being said, I think that this context is actually one of the few good use-cases for closed-source software, and the best way to ensure a robust hash algorithm is to keep the algorithm closed-source and limit its access to trusted parties.

4. Adversarial ML research already has a serious reproducibility problem. Most papers that adversarially test defensive measures seem to write their own attack code to do so. This makes sense if a novel attack is also being developed, but becomes problematic if the relative performance between two or more solutions is to be assessed in the context of a known attack strategy, because accurate comparison requires that the two attacks are identical or at the very least quite similar. For those interested, [this paper](https://arxiv.org/pdf/2003.01690) covers the issue in more detail and is another inspiration behind the project. 

This repo implements Meta's PDQ algorithm and contains some original PDQ code too, which is licensed under the BSD license. You can find the original implementation [here](https://github.com/facebook/ThreatExchange/tree/main/pdq).

## Overview
Spectra is a comprehensive adversarial evaluation toolkit for perceptual hash algorithms. Inspired by [Foolbox](https://github.com/bethgelab/foolbox), Spectra makes it easy to benchmark the robustness of perceptual hash pipelines against black and white-box evasion-based attacks. It's designed to be easily extensible in order to accommodate new attack strategies and hash algorithms, and also supports a very flexible experiment orchestration pipeline that can be used for quick one-off tests or highly comprehensive experiments alike. The ultimate goal is to eventually expand the project into a generic adversarial evaluation framework similar to Foolbox but with a higher degree of control and customizability over the orchestration pipeline and a particular focus on zeroth-order black-box attacks and non-differentiable optimization landscapes. However, for now it's mainly focused on perceptual image hashes, because that was the lowest-overhead pipeline to implement first, take it or leave it (or help improve it).


## How To Use

First clone the repo, get the submodule(s), and install the required dependencies. Afterwards, try running an example experiment through the `run_attacks.py` script like so:

```bash
git clone https://github.com/nolanmccleary/Spectra.git
./update_submodules.sh
pip install -r requirements.txt
python run_attacks.py -f pdq_attackmode_comparison.yaml
```


Or any other experiment you desire. Play around with the flags too and see how they change the image distortions. You can also write your own experiments/attacks, which is the primary purpose of this project. Please note that depending on your hardware, some experiments may take a while to run. 


You can also run multiple experiments at the same time, but do note that in doing so, any overrides will be applied across all attacks inside all experiments (100% done to make mass overrides easier and save time typing, not because I'm too lazy to rework it). For example:

```bash
python run_attacks.py -f ahash_experiment dhash_experiment -hamming_threshold 22
```

Will run both `ahash_experiment` and `dhash_experiment` and will force the hamming threshold across all attacks inside these experiments to 22. 



## Runtime Configuration

You can look inside spectra/config/attack_config.py to view the configuration scheme for experiments and attacks. Very briefly, Spectra is designed to primarily execute at the experiment level. Experiments contain one or more attacks, and also specify the target files used by all attacks that make up the experiment. All other attack configuration info is stored within the attack configuration, and `run_attacks.py` provides overrides for them as well. Please note that in its current state, these overrides are applied throughout the entire scope of the experiment, so explicitly setting the individual experiment configurations is recommended when running larger tests. 

Experiment and attack configurations can be set in three separate ways: 
1. From a YAML file 
2. From a dictionary
3. Directly through the constructor

For most cases, it's generally recommended to use either approach 1 or 3.

The fields for experiment and attack configuration are resolved at the time of experiment execution, meaning that initializing configs with incomplete field data is permitted provided that all fields have been filled appropriately when the experiment is actually run. Some fields also have defaults (i.e. default verbosity is low), and will resolve to them if the field is not explicitly configured or overridden. 


## Architecture

Spectra composes attacks under a few key assumptions:

1. We have an input tensor.
2. We have a function that operates on that input tensor and returns an output tensor. We are trying to break this function, even if we don't know the specifics of how it works. 
3. We can define a loss function that quantifies how well some given adversarial tensor is 'breaking' our target function output with respect to our input tensor. We can use this loss function to figure out the ideal way to adversarially perturb our input tensor. 
4. Optionally, we can also define an acceptance function that can decide on whether or not the target function has been 'broken'. If we don't define one, the optimization process will continue until some hard cutoff on the number of cycles is reached. 

The output of this optimization process is an adversarial delta tensor that, when combined with our input tensor, 'breaks' our target function in whatever context we define 'breaking' to be. This delta may also be post-processed according to the attack configuration's settings and added to a copy of the original image to create the adversarial image tensor. The process of generating that delta is performed by an Optimizer instance. Multiple instances are already present in order to facilitate a few different attack strategies (mainly for the sake of comparison) but the stanadardized interface makes it easy to add more depending on what sort of behaviour is desired. Strictly speaking, as long as there is some sort of loss function to define the optimization criteria, the `NES_Optimizer` should work relatively well in non-differentiable contexts. It's clearly overkill in a differentiable situation, but at that point just use Autograd lol. 

The combination of an NES-based optimizer in particular in conjunction with arbitrarily defined acceptance and loss functions offers a few key advantages:

1. NES does not require the function to be known. It also doesn't require the function to be differentiable. Thus, this approach allows us to develop tooling to approximate the gradients of black-box, non-differentiable functions (such as perceptual image hashes) provided we know the output tensor for any given input tensor.
2. Arbitrarily defined loss functions should make it easy to implement other attack strategies in the future (such as Carlini-Wagner collisions). This should be done sometime in the future.
3. Arbitrarily defined acceptance functions provide a flexible way to determine if some given target is adversarially sufficient for the test at hand. 

As mentioned previously, this approach is designed to be extensible beyond the scope of perceptual image hashes to accommodate other classification systems such as model-based classifiers as well as other forms of media such as audio files.


## Interesting observations

There have been two interesting observations made while developing this system. 

1. Attacks on PDQ barely affect ahash outputs. For example, when `pdq_experiment` is run with default settings we get the following:

```json
"average_results": {
  "average_ahash_hamming_torch": 2.185185185185185,
  "average_dhash_hamming_torch": 4.518518518518518,
  "average_phash_hamming_torch": 9.25925925925926,
  "average_pdq_hamming_torch": 56.592592592592595,
  "average_ahash_hamming_cannonical": 2.111111111111111,
  "average_dhash_hamming_cannonical": 7.851851851851852,
  "average_pdq_hamming_cannonical": 60.0,
  "average_phash_hamming_cannonical": 10.148148148148149,
  "average_ahash_discrepency": 0.07407407407407407,
  "average_dhash_discrepency": -3.3333333333333335,
  "average_phash_discrepency": -0.8888888888888888,
  "average_pdq_discrepency": -3.4074074074074074,
  "average_lpips": 0.028861979229582682,
  "average_l2": 0.05861155157563863,
  "average_ideal_step_coeff": 0.010000000000000002,
  "average_ideal_alpha": 2.9,
  "average_ideal_beta": 1.0,
  "average_ideal_scale_factor": 1.0,
  "average_num_steps": 31.444444444444443
}
```


2. A reasonably effective attack can be performed by just getting the initial gradient vector and stretching it until the hamming threshold breaks. This works better for some hashes than others, namely ahash and dhash. Another interesting observation is that this attack seems to better approximate a full NES attack when run on PDQ versus when run on PHash, meaning that the cosine similarity between colinear and NES deltas is higher during PDQ attacks than during PHash attacks at a given hamming threshold (adjusted for the hash size difference between the two). I don't know if this would imply that PHash is technically a bit more adversarially robust than PDQ because the optimizer is required to change direction a little bit more on each step, but it's an interesting thing to see regardless. I should note that so far I have only evaluated this in a very non-rigorous manner, so I may be missing something obvious here.


Through my extensive research (Perplexity and ArXiv binge sessions), I have not been able to find any information on these things in the current literature. However, as you should expect by now, my review has probably been less than thorough, and I may be missing something obvious. Perplexity and ChatGPT also claim that my attack results (from the configuration inside `full_attack_suite.yaml`) are hitting state-of-the-art numbers across all 4 hash algorithms tested. However, since there is virtually no standardization across attack statistics in the current literature, it's kind of anyone's guess as to what state-of-the-art even means here, and is yet another justification for why a standardized evaluation platform like this could be a good idea.




## Contributions

If you have a contribution that you think would improve the project, please open up a pull request. Some things that would be good to implement at some point:

1. More loss function options to support different attack objectives beyond simple gradient ascent. For example, a Carlini Wagner optimization mode. 
2. Smarter override logic in run_attacks.py such that you can bind overrides at the level of an individual attack or experiment. As I mentioned before, right now they are applied across all experiments, which is kind of dumb.
3. Add an option to average across num_reps when sweeping for ideal hyperparameters such that outlier runs don't skew results.
4. Randomized perturbation seeding such that more unique deltas are generated between any two runs.
5. A smarter hyperparameter sweep algorithm.
6. Extend the system to work with other forms of media e.g. audio files or other types of classifiers e.g. neural networks.
7. Run on a real image set and see how the attacks generalize i.e. ImageNet 1000.
8. Add an option to generate an attack artifact that stores all calculated delta tensors. Useful for training some sort of attack network in the future. This would be a really cool project and nobody has done it yet as far as I know.


I might get around to implementing some of these at some point, but at the moment I'm kinda sick of working on this thing, and I have a way more exciting (and probably more useful) project on the radar that I really wanna get started with. 


## Citations

If you use this system in your research, please cite this repo:

```bibtex
@misc{mccleary2025spectra,
  title={SPECTRA: Spectral Perceptual Experimentation and Cross-Testing Research Adversary},
  author={McCleary, Nolan},
  year={2025},
  howpublished={\url{https://github.com/nolanmccleary/Spectra}},
  note={A comprehensive adversarial evaluation toolkit for perceptual hash algorithms}
}
```

## Related Work

This project was based on the following work:

**Adversarial Detection Avoidance Attacks (Using NES):**
```bibtex
@inproceedings{jain2022adversarial,
  title={Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning},
  author={Jain, Shubham and Crețu, Ana-Maria and de Montjoye, Yves-Alexandre},
  booktitle={31st USENIX Security Symposium (USENIX Security 22)},
  pages={1--18},
  year={2022},
  url={https://www.usenix.org/system/files/sec22-jain.pdf}
}
```


**Natural Evolution Strategies (NES):**
```bibtex
@article{wierstra2014natural,
  title={Natural evolution strategies},
  author={Wierstra, Daan and Schaul, Tom and Glasmachers, Tobias and Sun, Yi and Peters, Jan and Schmidhuber, J{\"u}rgen},
  journal={Journal of Machine Learning Research},
  volume=15,
  number={27},
  pages={949--1009},
  year={2014},
  url={https://jmlr.org/papers/v15/wierstra14a.html}
}
```

**PDQ Hash Algorithm:**
```bibtex
@inproceedings{roy2020pdq,
  title={PDQ: Pretty darn quick perceptual image hashing},
  author={Roy, Avik and Sun, Oliver and Tunnell, James and Flynn, Patrick J and Bowyer, Kevin W},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={168--169},
  year={2020}
}
```

**LPIPS Perceptual Similarity:**
```bibtex
@inproceedings{zhang2018unreasonable,
  title={The unreasonable effectiveness of deep features as a perceptual metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={586--595},
  year={2018},
  url={https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf}
}
```

**Perceptual Hash Function Benchmarking:**
```bibtex
@article{zauner2010implementation,
  title={Implementation and benchmarking of perceptual image hash functions},
  author={Zauner, Christoph},
  journal={Master's thesis, Upper Austria University of Applied Sciences, Hagenberg Campus},
  year={2010}
}
```

**Adversarial ML Reproducibility:**
```bibtex
@article{engstrom2020adversarial,
  title={Adversarial robustness as a prior for learned representations},
  author={Engstrom, Logan and Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Tran, Brandon and Madry, Aleksander},
  journal={arXiv preprint arXiv:2003.01690},
  year={2020},
  url={https://arxiv.org/pdf/2003.01690}
}
```


**Foolbox (Inspiration):**
```bibtex
@article{rauber2017foolbox,
  title={Foolbox: A Python toolbox to benchmark the robustness of machine learning models},
  author={Rauber, Jonas and Brendel, Wieland and Bethge, Matthias},
  journal={arXiv preprint arXiv:1707.04131},
  year={2017},
  url={https://arxiv.org/pdf/1707.04131}
}
```

**Adversarial Attacks on Perceptual Hashes:**
```bibtex
@inproceedings{wang2019adversarial,
  title={Adversarial examples for perceptual image hashing},
  author={Wang, Shengshan and Zhang, Xiaoyu and Wang, Yifan and Liu, Ming and Wang, Leo Yu and Zhang, Shengchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9310--9319},
  year={2019}
}
```
