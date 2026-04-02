import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import pandas as pd
import time

from vqe_usecase import run_vqe_usecase, explain_results_vqe
from qaoa_usecase import run_qaoa_usecase, explain_qaoa_results, measure_bruteforce_times, get_demo_graph, plot_init
from gan_usecase import (run_usecase, plot_before, plot_after_q, plot_after_c, plot_distances, explain_results,)

st.set_page_config(layout="wide")
st.sidebar.title("Quantum Playground")

page = st.sidebar.radio(
    " ",
    ["Introduction",
     "Variational Quantum Eigensolver",
     "Quantum Approximate Optimization Algorithm",
     "Quantum Generative Adversarial Networks",
     "Hardware Arena",
     "Conclusion"]
)

if page == "Introduction":

    st.title("My Quantum Computing Use Case Journey")

    st.markdown("""
        This project is my exploration of practical industry use cases for quantum computing. 
        As we've all heard before quantum computing (QC) has the potential to impact almost every industry. 
        However, understanding how your company specifically is affected is less clear.
        My aim is to help tailor quantum computing to your companies individual needs.
        In order to achieve this, I delevoped the "Qantum Playground", where I have displayed what I have learned in the last couple of days and make it more enganging by adding small games to each section.
        Before we start, I want to mention that right now QC is still in an exploration stage. 
        Algorithms are still being developed and implementation and hardware are constantly evolving.
        Although, we are not at a stage where QC can compete with classical hardware, there are still key aspects of QC that should be considered:
        1. It is important to know that the quantum landscape is constantly evolving
        2. Once everything is ready, it is important to already be confident using it and having some experience in the company
        3. Especially in the beginning rare talent is required, the earlier a company acquires these talents or builds up skills, the better.
        
        Additionally, it is important to note that quantum advantage could take on a variety of different shapes, including: fewer required samples, a better solution quality, higher accuracy in simulation, maybe even a solution to problems that have not yet been solved, or - and thats something fundamentally different - a better energy efficiency.
        None of these advantages have been proven yet, but there are signs that we are on the right track.
        
        Now let's turn to the use cases, beginning with different industries. I have identified six areas where QC can play a role:
        1. Finance
        2. Logistics
        3. Healthcare
        4. Materials
        5. Cryptoanalysis
        6. Research

        For each area, key use cases have been identified although it is by no means a comprehensive list. The use cases are as follows:
    """)
    st.image("pic1.jpg", caption="Impacted areas with use cases")
    st.markdown("""
        One important insight is that we do not need a completely new approach for every individual use case. Instead, a small number of core algorithms and models can be applied across many scenarios.
        In this project, three key algorithms are explored further, as they consistently appear across the majority of use cases. Together, they cover the three main computational areas: optimization, simulation, and machine learning:
        1. Variational Quantum Eigensolver (VQE)
        2. Quantum Approximate Optimization Algorithm (QAOA)
        3. Quantum Generative Adversarial Networks (QGANs)
    """)
    st.image("pic2.jpg", caption="Underlying algorithms")
    st.markdown("""               
        As a starting point everything that can be described as a Hamiltonian is a good start for QC. 
        These kind of problems are natural for a quantum device and, therefore, show a lot of potential for an advantage.
        Due to a lot of other challenges and, frankly, very good classical algorithms, the goal is not to show advantage over the classical solution, but to show future potential of quantum solutions.

        These algorithms will be explored in more detail in the following pages. Afterwards, I will examine how different hardware architectures influence our solutions, and whether certain architectures are better suited for specific use cases.
        At the end of each page, there is a small game to check if you understood the section. Let's get into it!
    """)

elif page == "Variational Quantum Eigensolver":
    import vqe
    vqe.render_vqe_intro()
    

    if "vqe_results" not in st.session_state:
        st.session_state.vqe_results = None
    if "vqe_last_params" not in st.session_state:
        st.session_state.vqe_last_params = None

    steps = st.slider("Optimization steps", 5, 60, 25)
    learning_rate = st.selectbox("Learning rate", [0.05, 0.1, 0.2, 0.3], index=2)
    start_theta = st.slider("Starting parameter", -3.14, 3.14, 0.0)

    current_params = (steps, learning_rate, start_theta)

    # clear old results when inputs change
    if st.session_state.vqe_last_params != current_params:
        st.session_state.vqe_results = None

    submitted = st.button("Run game")

    if submitted:
        st.session_state.vqe_results = run_vqe_usecase(steps, learning_rate, start_theta)
        st.session_state.vqe_last_params = current_params

    results = st.session_state.vqe_results

    st.subheader("Results")

    if results is None:
        st.info("Choose settings and click Run game.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Quantum (VQE)", f"{results['quantum_energy']:.6f}")
        col2.metric("Classical (Exact)", f"{results['exact_energy']:.6f}")
        col3.metric("Difference", f"{results['error']:.6f}")

        st.info(explain_results_vqe(results["error"]))

        chart_df = pd.DataFrame({
            "Step": range(1, len(results["energy_history"]) + 1),
            "VQE energy": results["energy_history"],
            "Exact": [results["exact_energy"]] * len(results["energy_history"]),
        }).set_index("Step")

        st.subheader("Optimization progress")
        st.line_chart(chart_df, use_container_width=True)

    st.markdown("""
    ### Summary
    From this, it can be seen that by shifting from a classical to a hybrid algorithm, the dependencies of the computational cost change. With the hybrid algorithm, complexity can be reduced and, therefore, computational cost.
    This could lead to a more efficient or more accurate solution in the future.
    These problems are interesting for industries as well, because often a small time or quality improvement can have a large impact on the business outcome.
    Especially in the fields of drug or new material discovery this could improve the understanding of the material's behaviour.
    In this example, classical solutions can be done exact, which does not work for larger problems.
    The comparison of a classical variational algorithm with the VQE would be of interest for larger molecules.
    The solution could become faster, but it depends largely on the initial state and problem formulation.
    Additionally, there are a lot of other computational costs that need to be taken into account.
    Nonetheless, the algorithm works and comes closer to the ground state.
    """)

elif page == "Quantum Approximate Optimization Algorithm":
    import qaoa
    qaoa.render_qaoa_intro()

    if "qaoa_results" not in st.session_state:
        st.session_state.qaoa_results = None

    st.subheader("Why classical methods hit a limit")
    st.write(
        """
    For small graphs with a small amout of nodes, a classical computer can check every possible solution exactly.

    But the number of possible solutions grows as **2^n**.
    That exponential growth is why approximate methods like QAOA become interesting on larger problems.
    """
    )

    timing_rows = measure_bruteforce_times()
    timing_df = pd.DataFrame(timing_rows)
    timing_df["search_space"] = timing_df["search_space"].map(lambda x: f"{x:,}")
    timing_df["time_seconds"] = timing_df["time_seconds"].map(lambda x: f"{x:.6f}")

    st.dataframe(
        timing_df.rename(
            columns={
                "nodes": "Nodes",
                "search_space": "Classical search space",
                "time_seconds": "Brute-force time (s)",
                "best_cut": "Best cut",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Game graph")

    n_wires_demo, edges_demo = get_demo_graph()
    initial_fig = plot_init(
        n_wires_demo,
        edges_demo,
        title="Initial Traffic Graph"
    )
    st.pyplot(initial_fig)

    with st.form("qaoa_form"):
        p = st.slider("QAOA layers", 1, 3, 1)
        steps = st.slider("Optimization steps", 5, 50, 20)
        learning_rate = st.selectbox("Learning rate", [0.1, 0.3, 0.5, 0.7], index=2)
        shots = st.selectbox("Samples", [50, 100, 200], index=1)
        submitted = st.form_submit_button("Run game")

    if submitted:
        st.session_state.qaoa_results = run_qaoa_usecase(
            p=p,
            steps=steps,
            learning_rate=learning_rate,
            shots=shots,
        )

    results = st.session_state.qaoa_results

    st.subheader("Results")

    if not submitted and results is None:
        st.info("Choose settings and click Run game.")
    elif not submitted:
        # 👇 prevents old results from showing automatically
        st.info("Click 'Run game' to generate results.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Classical best cut", f"{results['classical_best_cut']:.2f}")
        col2.metric("QAOA expected cut", f"{results['qaoa_expected_cut']:.2f}")
        col3.metric("Quality", f"{results['approximation_ratio']:.2f}")

        st.info(explain_qaoa_results(results))

        chart_df = pd.DataFrame({
            "Step": range(1, len(results["expected_cut_history"]) + 1),
            "QAOA expected cut": results["expected_cut_history"],
            "Classical optimum": [results["classical_best_cut"]] * len(results["expected_cut_history"]),
        }).set_index("Step")

        st.subheader("Optimization progress")
        st.line_chart(chart_df, use_container_width=True)

        st.subheader("Graph Visualization")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.pyplot(results["initial_fig"])

        with col2:
            st.pyplot(results["classical_fig"])

        with col3:
            st.pyplot(results["qaoa_fig"])


    st.markdown("""
    ### Summary
    We can see in this example, it is possible to find a solution with our quantum hybrid method.
    As mentioned before, the system size is so small that we can have a classical exact solution. 
    As shown in the scaling behaviour approximate solutions are required for larger problems.
    To make a quantum advantage consideration, the comparison to a classical approximate solution would be required.
    Currently, the computational overhead for the quantum solution would likely overcome the potential benefit.
    Similarly to the VQE part, it also depends on the problem formulation and is difficult to quantify.
    """)

elif page == "Quantum Generative Adversarial Networks":
    import gan
    gan.render_gan_intro()

    if "demand_results" not in st.session_state:
        st.session_state.demand_results = None

    st.subheader("Use case: daily demand simulation")
    st.write("A small shop wants to simulate daily customer demand. We compare a quantum and a classical model.")

    with st.form("demand_form"):
        steps = st.slider("Training steps", 3, 20, 10)
        lr = st.selectbox("Learning rate", [0.05, 0.1, 0.15, 0.2], index=2)
        seed = st.number_input("Seed", 0, 999, 0)
        submitted = st.form_submit_button("Run game")

    if submitted:
        st.session_state.demand_results = run_usecase(steps, lr, seed)

    results = st.session_state.demand_results

    st.subheader("Results")

    if not submitted and results is None:
        st.info("Choose settings and click Run game.")
    elif not submitted:
        st.info("Click 'Run game' to generate results.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Quantum distance", f"{results['q_after_dist']:.3f}")
        col2.metric("Classical distance", f"{results['c_after_dist']:.3f}")

        st.info(explain_results(results))

        st.pyplot(plot_before(results))

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_after_q(results))
        with col2:
            st.pyplot(plot_after_c(results))

        st.pyplot(plot_distances(results))

    st.markdown("""
    ### Summary
    It can be seen that the used algorithm does in fact train the generator to a certain data set.
    In this small toy problem with a weak classical component the qGAN can actually compete with the classical GAN.
    In general, in quantum machine learning there are four different models that have to be investigated individually and also have different advantages and disatvantages:
    1. classical input, classical output 
    2. classical input, quantum output 
    3. quantum input, classical output
    4. quantum input, quantum output
    
    With each type there are different challenges related to converting and loading the data into and out of the model. 
    A lot of these models would need a functioning qRAM to work, which are not available yet.
    As a workaround, the computation has to be done in the qubit's coherence time which may be too short for the algorithms.
    Additionally a lot of sampling is required which is hindering the potential advantage once again.
    Overall, there are a few cases where quantum could have a positive impact. However, the impact is dependent on the problem formulation and its components.
    Especially when considering the combination with quantum data from quantum sensor for example, qml could play a role in the future.
    """)

elif page == "Hardware Arena":

    st.title("Quantum Hardware Arena")

    st.markdown("""
### Hardware Approaches
Quantum Computing involves using quantum mechanical phenomenons to perform computational tasks.
For that something that can work like a qubit is required. This means a quantum system must have two states that can manipulated.
A set of operators is necessary to manipulate the state of a qubit in a certain way, so computational operations can be performed.
The operations on one or more qubits are called gates. They have two different parameters that matter for quantum computing:
                
a. Their 'quality', also called fidelity: how accurate is the result of the operation. The closer to 100% the less errors our algorithm gets and the less correction is required

b. Their speed: the time the operation takes matters in the duration of the overall algorithm. This is also an important parameter when we think about the stability of a qubit.

The underlying properties can be achieved in many different ways and I want to quickly introduce the most common ones and give a quick overview of their current properties and challenges.
The architectures are:
1. Ion Traps
2. Neutral Atoms
3. Superconducting 
4. NV Centers
5. Photonic
6. Spin
7. Topological
""")    
    data = [
        {
            "Architecture": "Superconducting",
            "Current size": "100 – 1000+ qubits",
            "Advantages": "Fast gates, mature tech",
            "Challenges": "Short coherence, scaling wiring",
            "Companies": "IBM, Google, Rigetti"
        },
        {
            "Architecture": "Ion Trap",
            "Current size": "20 – 100 qubits",
            "Advantages": "Very high fidelity",
            "Challenges": "Slow gates, scaling complexity",
            "Companies": "IonQ, Quantinuum"
        },
        {
            "Architecture": "Neutral Atoms",
            "Current size": "100 – 1000 qubits",
            "Advantages": "Scalable arrays",
            "Challenges": "Gate precision, control",
            "Companies": "PlanQC, QuEra, Pasqal"
        },
        {
            "Architecture": "Photonic",
            "Current size": "10 – 100 modes",
            "Advantages": "Room temperature, networking",
            "Challenges": "Loss, probabilistic gates",
            "Companies": "Xanadu, PsiQuantum"
        },
        {
            "Architecture": "Spin",
            "Current size": "10 – 100 qubits",
            "Advantages": "CMOS compatibility",
            "Challenges": "Control precision",
            "Companies": "Intel, Silicon Quantum Computing"
        },
        {
            "Architecture": "NV Centers",
            "Current size": "2 – 10 qubits",
            "Advantages": "Long coherence",
            "Challenges": "Scaling entanglement",
            "Companies": "SaxonQ, XeedQ"
        },
        {
            "Architecture": "Topological (early)",
            "Current size": "experimental",
            "Advantages": "Error-resistant (theoretical)",
            "Challenges": "Not yet realized",
            "Companies": "Microsoft"
        },
    ]

    df = pd.DataFrame(data)

    st.subheader("Current hardware landscape")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("""
### Ion Traps
Ion Trap quantum computers use ions, that they trap and cool down with Pauli-traps.
This leads to ions that can be manipulated by lasers or microwaves.
With this technique the qubits are very stable and, due to the fact they can be moved, all-to-all connected.
The gate fidelity is very high compared to other architectures with a leading 99.99% two-qubit fidelity achieved by IonQ.
The overall speed of the gates is low. 
Additionally, the shutteling, the movement to a spot where the qubits can be manipulated has to be very slow to prevent the ions from being heated and, therefore, disturbed.
Due to this as well as other variables the overall speed of computations in this architecture is slow.
The current challenge of ion trap quantum computers is the overall scaling. 
Bringing all required elements in an easily scalable and interactive way has not yet been solved.

### Neutral Atoms
Neutral atom quantum computers are similar to ion traps and use neutral atoms that are trapped and manipulated to act like a qubit.
However, they use optical components and Rydberg excitations to control the atoms while lasers are used to perform gates on them.
The scalability and connectivity are advantageous for these systems, while fidelities and precise controls are still a challenge.
A well known startup working on neutral atom quantum computing is planqc, based in Munich, Germany.
                
### Superconducting
Superconducting qubits are realized by small electrical circuits that are cooled to near absolute zero. 
At this temperature they behave quantum mechanically and realize qubits.
The gates are implemented by microwave pulses that control the current.
This architecture leads to very fast gate operations and can use well-developed technology. Currently these systems have the largest amount of qubits.
The fidelity and the coherence time is low compared to other architectures. The qubits are not all-to-all connected.
The scalling seems to become more and more difficult due to the cryogenic challenges.
                
### NV Centers
NV center quantum computing works with defects in a diamond.
Each NV center can contain many qubits. 
For an efficent scaling, a connection between an array of NV centers is required.
There are a few companies trying to build NV center QC that operate at room temperature and are, therefore, very adaptable and applicable for a few specific use cases.
Challenges surrounding the scaling of the system still persist.
NV centers need to be in a specific distance and with a certain direction. And until it can be done it is not scalable because it is an almost arbitrary process.
However, the architecture shows a lot of potential for quantum sensing.
                
### Photonic
Photonic QC are unique compared to the other architectures. 
In this architecture light and photons are used as qubits and the computational tasks are performed by beam splitters and interferometers.
This architecture can be performed at room temperature and is well fitted for communication and networking use cases.
The current challenges lie in the photon loss combined with single photon detection and creation. 


### Spin 
Spin qubits use the spin of electrons in semiconductor materials. 
The control of the qubits is done by small and precise electric and magnetic fields.
Eventhough they can make use of the existing chip manifacturing, precise controls are difficult to achieve.

                   
### Topological
Topological qubits are arguably the most unique QC architecture.
As it is not proven to even work as a qubit, the details are are not part of this section.
However it is improtant to note, because Microsoft claimed a working topological qubit, which was not well received by the community.
If topological qubits work, the scaling would be easy and is therefore, still a competitor in the quantum race.

""")    

    import streamlit as st
    import numpy as np

    st.title("Hardware Game")

    st.write("""
    Let's finish this section with a game as well.
    We want to see how well a quantum algorithm would work on a current architecture.
    Each architecture has its individual:
             
    - gate fidelity
    - speed
    - scalability
             
    We want to see which kind of algorithm would run well on which hardware.
    For that, we can choose one of the above hardware architectures (except topological) and define our algorithm by:
             
    1. Number of qubits: how many qubits do we need in our algorithm?
    2. Circuit depth: how deep is our algorithm, how many gates do we apply?
    
    Let's run some algorithms! :)
    """)


    hardware = {
        "Superconducting": {
            "fidelity": 0.995,
            "max_qubits": 1000,
            "gate_time": 1,
        },
        "Ion Trap": {
            "fidelity": 0.9999,
            "max_qubits": 100,
            "gate_time": 50,
        },
        "Neutral Atoms": {
            "fidelity": 0.99,
            "max_qubits": 1000,
            "gate_time": 5,
        },
        "Photonic": {
            "fidelity": 0.97,
            "max_qubits": 200,
            "gate_time": 1,
        },
        "Spin": {
            "fidelity": 0.995,
            "max_qubits": 200,
            "gate_time": 10,
        },
    }

    # Inputs

    hw_choice = st.selectbox("Hardware", list(hardware.keys()))
    qubits = st.slider("Number of qubits", 5, 200, 50)
    depth = st.slider("Circuit depth", 1, 200, 50)

    run = st.button("Run algorithm")

    # Code

    if run:
        hw = hardware[hw_choice]

        # check hardware limit
        if qubits > hw["max_qubits"]:
            st.error("Too many qubits for this hardware!")
        else:
            fidelity = hw["fidelity"]
            gate_time = hw["gate_time"]

            gates = qubits * depth
            success = fidelity ** gates
            runtime = gates * gate_time
            score = success / np.log(runtime + 1)

            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Success probability", f"{success:.6f}")
            col2.metric("Runtime (relative)", f"{runtime:.0f}")
            col3.metric("Score", f"{score:.6f}")

            # interpretation
            if score > 0.05:
                st.success("Good balance between speed and accuracy")
            elif score > 0.01:
                st.info("Decent trade-off")
            else:
                st.warning("Poor performance")

            st.write(f"""
    You used **{hw_choice}**. 
    Total operations: **{gates}**. 
    Each operation takes **{gate_time} time units**. 
    """)
    
elif page == "Conclusion":

    st.title("Conclusion")

    st.markdown("""
    I have built this app as an attempt to bridge the gap between professionals with a quantum background and business professionals trying to understand algorithms. 
    This is a topic I personally have overcome having both a quantum and business background. 
    With the use of small interactive problems, I wanted to showcase three algorithms that can play a huge role in the future of quantum computing.
    Of course there are many more use cases and many variations of the algorithms we have explored here and much more to learn. Addtitionally, as the hardware is evolving so too is software development.
    There are many companies and products already on the market that simplify working on these use cases, where the user/developer requires less quantum knowledge.
    My goal was to show that it is possible to go into detail without negating a business element (and while trying to have fun).
    I believe there is still a large gap between interested businesses and quantum computing due to a lack of understanding.
    It is my hope that this fun interactive approach allows a wider audience to better grasp the question "What is Quantum Computing" without comprehensively covering all aspects of QC.
    In the future, I would love to integrate real world hardware into my quantum playground and see how available NISQ devices perform, where the overhead comes from, and when an advantage can be seen.
    Lastly, I would like to incorporate more use cases and dive even further into a variety of interesting topics with more detailed coding.
    """)