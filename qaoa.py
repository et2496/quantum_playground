import streamlit as st

def render_qaoa_intro():
    st.title("Quantum Approximate Optimization Algorithm (QAOA)")

    st.write("""
        ### Basics
        The QAOA is an algorithm to solve binary decisions. For example, in logistics, a routing problem is a set of binary decisions: where do I go next?
        The solution space grows here exponentially with the number of locations.
        This problem has the following dimensions:
        1. Problem size: how many locaions are part of the problem? These are called nodes.
        2. Edges: the edges represent the traffic between the nodes and, therefore, the cost.
        3. Constraints: are there any constraints that must be followed when travelling?
        4. Time: are there time constraints?
        5. Solution type: is a global minimum reached or is it a local minimum?

        ### Classical Limits
        These four dimension already show the classical limitations. 
        As mentioned before the solution space grows exponentially with size. The constraints could also impact the solution quality or time.
        All in all with classical methods, we quickly get to a point where we can't get a global minimum for the exact system in a given time.
        A decision then has to be made: Do I accept the best solution at that time? Do I wait longer for the solution? Do I make assumptions to the system to decrease solution time?
                    
        ### Quantum Element
        This is why quantum solutions should be considered in this case. 
        It is a common problem in a variety processes and industries. Everyday, companies have to make one of the above compromises.
        So what could we change when we switch from a classical solution to a hybrid quantum solution?
        We describe the system with problem Hamiltonian and a mixer Hamiltonian. We then use a classical optimizer to improve the solution similar to our VQE optimizer.
        This can be beneficial because the problem Hamiltonian contains many pairwise couplings and is, therefore, more easily solvable.
        Before we look at an example, it is important to say that trapped ion architectures could potentially benefit from their all-to-all qubit connection that decreases the circuit depths and ultimately decreases the accumulated error.

        ### Example
        A logistics company that wants to split their route into two operating zones. This is called a MaxCut problem. We try to find the best split for the network.
        In this example, we can directly see why we need other algorithms besides the brute force algorithm. 
        The brute force algorthm scales exponatially with the number of nodes and, therefore, is not applicable for larger systems.
        There are approximate solutions for this MaxCut problem that are better than the hybrid solution, but here only the QAOA is analyzed.
        Of course for a good benchmarking we would need the best classical approximate solution.
        The hope for quantum solutions is that quantum can represent a superposition of many solutions at once and, hence, solve it quicker.
        The example should showcase that the QAOA algorithm works. The following parameters can be adjusted to squeeze our solution:
        1. QAOA layers: how often do we apply our quantum gates in each algorithm step?
        2. Optimization steps: how often do we adjust our parameters?
        3. Learning rate: how much do we change our parameters in each step?
        4. Samples: how many samples do we get to find the quantum result?
        
        Let's try to find the optimal solution!
    """)
