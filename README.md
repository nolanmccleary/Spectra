# Adversarially Testing Perceptual Hashes


## Disclaimer

There are a lot of very well-justified use-cases for perceptual hash algorithms. I am not trying to compromise them. All things considered, I believe that open-sourcing my code/algorithm is justified due to the following reasons. 

1. There is a lot of published research on gradient-based evasion attacks dating back to around 2021. The algorithms for most of these attacks are visible in their respective papers. This means that platforms have had more than enough time to prepare accordingly, and it is very likely that most if not all of them have, even if it is not publicly disclosed.  

2. I am not providing a turn-key solution. If someone is technically competent enough to reverse-engineer my system and deploy it at scale, they are also most likely competent enough to build a similar attack of their own if they were so inclined.

3. Good image hashing pipelines do and should use multiple different hashes. The current system can only target one hash at a time. If an image recognition pipeline can be effectively evaded by this system, then that pipeline shouldn't have existed in the first place.

4. If an attacker really wanted to, they could achieve similar hash deltas to my system by carefully scribbling on the images and comparing the hashes. Obviously the distortion would be higher, but the harm factor of the content, if it were harmful, would remain the same. 

5. The benefits gained by improved transparency and reproducibility outweigh the drawbacks.



## Motivation

Perceptual hash algorithms are a widely used form of image detection, wherein a special algorithm is used to encode core image features in an n-bit hash, where n is variable depending on the algorithm context. This allows for two important attributes to be obtained:

1. An image 'fingerprint' can be acquired, which is much more convenient to store than the actual image. 

2. The 'fingerprint' encoding is made from the core features of the image, and is thus resistant to minor changes. 

This makes perceptual hashes widely used tools for the detection of copyrighted material or harmful content. However, such a system naturally begs the following question:
"Is there a way in which we can perturb the image such that the hash difference is greatest for some perturbation of a given magnitude?"

The answer is yes, to a different degree depending on the hash algorithm used. The motivation behind this project is two-fold:

1. Determine which perceptual hash algorithms are more or less resistant to gradient-based evasion attacks. 

2. Determine which perceptual hash algorithms are most complementary to each other when deployed in an image recognition pipeline.


There is [published research](https://arxiv.org/pdf/2106.09820) that implictly touches on these two points, but it quantifies attack/hash effectiveness by the rate at which false positive/false negatives occur in a pipeline over a database of a given size with n poisoned images. For content moderation systems, I think this makes sense. However, I think that it would be more useful to first broadly quantify how succeptible a given hash is to a basic gradient-based attack. To my knowledge, no published research clearly spells this out in a practical manner. Perceptual hash algorithms tend to be non-differentiable, however the difficulty of performing zeroth-order approximation on a given algorithm does vary. This will be clarified in the following section.


## How a Gradient-Based Attack Works (NES Flavour)

Perceptual hash functions tend to be non-differentiable. To think about this, imagine the image as a point in high-dimensional space, where each axis of travel represents the pixel intensity for a given pixel on the image. As we move our point around in high-dimensional space, the hash value will change accordingly. However, since hash functions are non-differentiable, the function relating the hash value to the point's position is not a continuous function. If we were trying to change the hash as much as possible while changing the position of our point as little as possible, we'd want to find the direction in which we could move the point the least while changing the hash the most. From multivariable calculus this is known as the gradient, and the gradient vector will point in the same direction as the ratio between how the hash changes with point shift across all axes of motion. 

However, because the hashes are non-differentiable, it is hard to determine the rate at which the hash will change when the point is moved in a given direction, thus it is also hard to compute the gradient. To get around-this, we can perform a linear approximation dH/ds (where H is the hash value and s is the magnitude of our position shift) across all (most) possible directions of motion in order to obtain an approximation of the gradient. Then adjust our point in the direction of that vector in order to change our hash value.


Since our image is a very high-dimensional vector, it's not practical to model change across every direction, so instead we will seek to maximize evenly-spaced directional coverage. In three-space, if we drew our target vectors coming out from our point, it would look like an [old-school naval mine](https://www.alamy.com/naval-mine-isolated-old-sea-mine-3d-rendering-3d-illustration-image465891455.html) (non-Ravikantian variety) where each vector is more or less evenly spaced from from the others. Instinctively we know that to do this, we should sample our vectors from a random distribution, but which distribution should we use? Fortunately, we also instinctivelly know that when in doubt, using the Gaussian distribution tends to work well, and that is indeed the distribution we will use here. 

One might ask why the Gaussian distribution will actually work the best here? Visually the reason why is because the gaussian distribution has a zero-mean and is symmetric. Zero-means that in our three-space point example, if each vector maps to a force exerted on our point, then given a lot of vectors the forces will cancel out and our point will not move. Symmetry means, in our three-space example, that if we enclose the point inside a sphere, we tend to have as many vectors on one hemisphere as the other, and the average magnitudes of the vectors inside either hemisphere will be equal. This means that the vector sum of the two hemispheres will be approximately equal in magnitude and opposite in direction, no matter how the sphere is split in half.

Technically, other zero-mean, symmetric distributions can be used in place of the Gaussian distribution. The reason why the Gaussian distribution is preferred comes down to one final factor known as rotational symmetry or isotrophy. Some distributions like the bernouli distribution are zero-mean and symmetric, but don't spread out the vectors nicely because it oversamples coordinate axes. This means that in three-space, our old-school naval mine has its spikes in clusters instead of spread out evenly across the surfaces. The net force exerted is still zero and all hemispherical vector sums still cancel each out, but we are not accurately capturing all possible directions of movement and thus our gradient estimation is prone to inaccuracy.

There are some other less obvious reasons (extreme samples are rare, Stein's Lemma can be used to simplify our math since our score function for the nudge vector is just the nudge vector itself, etc) but they are outside the scope of this document - except Stein's Lemma, which will be useful later.

Now we would like to know how we obtain this gradient. What we can do is scale each nudge vector by the magnitude of the hash delta that it incurrs, sum all scaled vectors, and then divide by the total number of vectors in order to get the average direction that corresponds with hash change. This vector points in the same direction as the vector representing the ratio between how the hash changes with point shift across all axes of motion, and thus will approximate our gradient. We have to scale each nudge vector appropriately before computing the hash though, if they are too small no hash change will occur and we can't compute the gradient. If they are too large, they won't accurately reflect gradient characteristics at that particular point. Tuning this scale factor is an involved process and varies depending on the hash being tested as the optimization surface will change accordingly.

This whole streategy of using normally distributed perturbations to estimate a gradient falls under a family of algorithms known as Natural Evolutionary Strategies (NES), and composes the core of our attack logic.




## License

This project is licensed under Restricted Research License with Ethical Usage Restrictions - see the [LICENSE](LICENSE) file for more details.

## Third-Party Components

This project includes the PDQ perceptual hashing algorithm, which is licensed under the BSD License by Meta Platforms, Inc. and affiliates. The PDQ implementation is used for research and educational purposes only, in accordance with both the BSD License and this project's Restricted Research License.

## Contact

For questions about ethical usage or responsible disclosure, please open an issue in this repository.