import streamlit as st

def render_vqe_intro():
    st.title("Variational Quantum Eigensolver (VQE)")
    st.markdown("""

### Basics
VQEs are mainly used in quantum chemistry and material science applications.
It's a hybrid algorithm that uses both quantum and classical components.
The main objective is usually to find the system's ground state and its corresponding energy.
This is done by expectation values of the Hamiltonian H. The quantum element is used to simulate the current state and calculate its according expectation value.
This expectation value is often called cost function.
The classical element is used as an optimizer to improve the state and its energy and usually minimize the energy and, therefore, the cost function.

### Classical Limits
With a classical computer and, hence, the classical ansatz the problems you can solve are limited. 
The Hamiltonian that is usually a matrix, encodes the size of the problem as well.
The size limit of the matrix depends on the matrix' properties:
If the matrix is sparse, the problem size can be large even with classical computers. When it does exceed a certain size or is less sparse, heuristics are required to diagonalize the matrix.
This means, the problem is solved approximately, not exactly.
The classical solution in a variational approach is reached by:
- An initial guess of the ground state
- Variation of the state to improve the initial guess
        
The process would look like:
1. Initial guess
2. Calculate expectation value
3. Alter the state in small steps
        
By repeating step 2 and 3, you learn about the landscape around your initial guess and see if you approach lower energies. The lower the energy the closer your solution is to the ground state and,
hence, to the ground state energy.

The amount of times the process has to be done, scales exponentially with the calculation of the expectation value which is computationally the most expensive step.
If QC could perform this step more efficiently, it could potentially have a large impact.
### Quantum Element
Instead of calculating the expectation value, we could estimate the state using a quantum computer with a variational ansatz and then measure the energy.
We can write the Hamiltonian as a linear combination of easily measureable terms and then measure each of these terms and get the expectation value as weighted sum over these.
This would lead us away from having large multiplications, as it is done in the classical way, to a measurement that uses Pauli operators to represent the same terms.
The scaling could be way more efficient than in the classical ansatz depending on the Pauli representation of the problem.
This quantum part and its results is then used in the classical variational approach and the state is varied to find information and at the end, the lowest energy and the ground state.
In this example, high measurement accuracy is therefore crucial, as lower accuracy would require more measurements and undermine the scaling advantage.

### Example
Here is a small game to show the optimization process. 
A hydrogen molecule was simulated to see how the VQE algorithm approaches the exact solution of the problem. 
The following parameters of the algorithm can be adapted:
            
1. Optimization Steps: How often do we go through our loop?
2. Learning Rate: Defines the step size of each loop.
3. Starting Parameter: Where do we start our optimization?
            
Feel free to play with these parameters to understand the problem.
The closer our VQE energy comes to the exact solution the better. 
""")