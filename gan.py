import streamlit as st

def render_gan_intro():
    st.title("Quantum Generative Adversarial Networks (qGANs)")

    st.markdown("""
    ### Basics
    In the last example, qGANs are explored.
    A classical GAN contains two parts: a generator and a discriminator. The generator learns a distribution and generates fake data from this learned distribution.
    The discriminator on the other hand tries to distinguish between the real data and the fake data from the generator.
    The generator and the discriminator get optimized to fake better data and accurately distinguish between the two data sets.
    In finance for example, many use cases depend on Monte-Carlo simulations that simulate the evolution of a given system. 
    This simulation takes a lot of samples to minimize its errors and, hence, takes a long time to evolve.
    
                    
    ### Classical Limits
    This highlights the classical limits of the use case. The simulation requires a lot of samples.
    It needs to run a relatively long period of time, to minimise the errors and get a proper solution.
    Especially GANs have a few risks in their learning process: instable learning, mode-collapse, or non-convergence.
    There are several classical attempts to improve the performance for different use cases and goals.
    
            
    ### Quantum Part
    Quantum Generative Adversarial Networks are the quantum alternative to classical GANs, where one or both parts can be performed on a quantum device.
    Especially changing the generator to a quantum generator seems to have an impact on the overall performance.
    Many use cases require a probability distribution. Fortunately, this can be represented by a quantum state of a quantum computer.
    The loading of this probability distribution only scales polynomial compared to the exponential scaling in the classical solution (see VQE).
    In this case, quantum solutions could offer a quadratic speed-up. 
    QGANs are again a hybrid solution that uses classical learning algorithms with a quantum state that evolves.
    
    
            
    ### Example
    In this example (q)GANs are used to estimate the demand of a shop. We learn the distribution of demand by quiet and busy days.
    We train the model on the realistic data and see how well it performs.
    Eventhough there would be plenty of different parameters to adjust, we keep it simple by only adjusting:
    1. Training steps: how often do we adjust our parameters?
    2. Learning rate: how much do we change our parameters in each step?
    3. Seed: Picking different random distributions.
                
    As a reference, there are several different architectures that could be used for the generators and the discriminator.
    We picked simple generators that have a comparable performance.
                
    Let's try to estimate the demand!
    """)