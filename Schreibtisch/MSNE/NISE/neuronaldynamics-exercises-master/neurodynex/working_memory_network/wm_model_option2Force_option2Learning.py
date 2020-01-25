"""
Implementation of a working memory model.
Literature:
Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and
network dynamics underlying spatial working memory in a cortical network model.
Cerebral Cortex, 10(9), 910-923.

Some parts of this implementation are inspired by material from
*Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013*,
online available.

Note: Most parameters differ from the original publication.
"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

import brian2 as b2
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from collections import deque
from neurodynex.tools import plot_tools
import numpy
import matplotlib.pyplot as plt
import math
from scipy.special import erf
from numpy.fft import rfft, irfft

b2.defaultclock.dt = 0.05 * b2.ms

#################################### HELPFUL FUNCTIONS###########################################

def decode(firing_rate,N_e):
    angles = numpy.arange(0,N_e)*2*numpy.pi/N_e
    R = []
    R = numpy.sum(numpy.dot(firing_rate,numpy.exp(1j*angles)))/N_e
    angle = numpy.angle(R)
    if angle < 0:
        angle +=2*numpy.pi 
    return angle #(angle,np.abs(R)) 


def get_results(files):
    """
    Gets results from a file for perturbed, non-perturbed motor trials
    Input:  filname_reg = path of unperturbed result file   n x 3   [cue angle, distractor angle, final bump position]
            filename_d  = path to perturbed results file    n x 3   [cue angle, distractor angle, final bump position]
    Output: 
            results of specified input
    """
    f=open(files)
    s=f.readlines()
    results=[]
    for i,s1 in enumerate(s):
        try:
            results.append([float(s1.split(" ")[0]),float(s1.split(" ")[1]),float(s1.split(" ")[2]),float(s1.split(" ")[3])])
        except:
            continue
        
    results=numpy.array(results)
    return results


################################### WORKING MEMORY ##############################################
def simulate_wm(
        N_excitatory=1024, N_inhibitory=256,
        N_extern_poisson=1000, poisson_firing_rate=1.4 * b2.Hz, weight_scaling_factor=2.,
        sigma_weight_profile=20., Jpos_excit2excit=1.6, adapt = True, error=180, prev_GEE=0*b2.nS, norepi=0,
        stimulus_center_deg=180, stimulus_width_deg=40, stimulus_strength=0.07 * b2.namp,
        t_stimulus_start=0 * b2.ms, t_stimulus_duration=0 * b2.ms,
        distractor_center_deg=90, distr_steps =30, distractor_width_deg=40, distractor_strength=0.0 * b2.namp,
        t_distractor_start=0 * b2.ms, t_distractor_duration=0 * b2.ms,
        learning_factor=0.002,
        G_inhib2inhib= 0.35*1.024 * b2.nS,#0.35
        G_inhib2excit= 0.35*1.336 * b2.nS,#0.35
        #G_excit2excit= 0.35*0.381 * b2.nS,#0.35
        G_excit2inhib= 0.35*1.2*0.292 * b2.nS,#1.2 *
        monitored_subset_size=1024, sim_time=800. * b2.ms):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +\- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    
################################# PARAMETER SETTINGS######################################################
    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction

    #
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = .9 * 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = .65 * 100.0 * b2.ms  # orig: 100
    tau_NMDA_x = .94 * 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    G_extern2inhib = 2.38*b2.nS
    G_extern2excit = 3.1*b2.nS

    # projectsions from the inhibitory populations
    G_inhib2inhib *= weight_scaling_factor
    G_inhib2excit *= weight_scaling_factor

    # projections from the excitatory population
    #G_excit2excit *= weight_scaling_factor 
    G_excit2inhib *= weight_scaling_factor  # todo: verify this scaling

#################################### STIMULUS SETTINGS ###################################################
    
    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    t_distractor_end = numpy.zeros(2*int(distr_steps/2))*b2.ms
    for t in range(0, 2*int(distr_steps/2)):
        t_distractor_end[t] = t_distractor_start[t] + t_distractor_duration
        #print(t_distractor_start[t])
        #print(t_distractor_duration)
    # compute the simulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]
    # compute the distractor index
    distr_width_idx = int(round(N_excitatory / 360. * distractor_width_deg / 2))
    
    #distr_center_idx = numpy.zeros(2*int(distr_steps/2)+1)
    #distr_target_idx = numpy.zeros(2*int(distr_steps/2)+1)
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_target_idx = [idx % N_excitatory for idx in range(distr_center_idx - distr_width_idx, distr_center_idx + distr_width_idx + 1)]
        
    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = \
        [(Jneg_excit2excit +
          (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * (360. * min(j, N_excitatory - j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
         for j in range(N_excitatory)]
    # validate the normalization condition: (360./N_excitatory)*sum(presyn_weight_kernel)/360.
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
    weight_profile_45 = deque(presyn_weight_kernel)
    rot_dist = int(round(len(weight_profile_45) / 8))
    weight_profile_45.rotate(rot_dist)

######################################### INHIBITORY POPULATION #########################################
    
    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                       size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

######################################## EXCITATORY POPULATION ##########################################
    
    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        G_excit2excit : siemens
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")
    
    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    
    excit_pop.G_excit2excit = 0.35*0.381*b2.nS*weight_scaling_factor 
    
    # form subgroup for center neurons
    excit_center = excit_pop[int(N_excitatory/2.-stimulus_width_deg/2.*N_excitatory/(360)):int(N_excitatory/2.+stimulus_width_deg/2.*N_excitatory/(360))]
    excit_shiftcenter = excit_pop[int(stimulus_center_deg/2.*N_excitatory/360.-distractor_center_deg*N_excitatory/360.-distractor_width_deg/2.*N_excitatory/(360)):int(stimulus_center_deg/2.*N_excitatory/360.-distractor_center_deg*N_excitatory/360.+distractor_width_deg/2.*N_excitatory/(360))]
    
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excit2inhib.connect(p=1.0)
    
    # set connections for changing GEE
    # syn_norepi2excit = Synapses(excit_center, excit_center, on_pre="s_NMDA_post += norepi")
    # syn_norepi2excit.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the STRUCTURED recurrent input. use a network_operation
    
######################################### NETWORK OPERATIONS #############################################

    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = numpy.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_total)
        excit_pop.s_NMDA_total_ = s_NMDA_tot

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            # excit_pop[stim_start_i - 15:stim_start_i + 15].I_stim = 0.25 * b2.namp
            # Todo: review indexing
            # print("stim on")
            excit_pop.I_stim[stim_target_idx] = stimulus_strength
        else:
            # print("stim off")
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        for k in range(0, 2*int(distr_steps/2)):
            if t >= t_distractor_start[k] and t < t_distractor_end[k]:
                excit_pop.I_stim[distr_target_idx] = distractor_strength[k]
                
    @network_operation(dt=1 *b2.ms)
    def norepinephrin(t):
        K = 1*b2.nS*(error-stimulus_center_deg)
        K_shift = 1*b2.nS*(-(stimulus_center_deg-distractor_center_deg))
        tau_NE = 200*b2.ms
        # define rise/fall of Norepinephrin
        if adapt==True:
            excit_center.G_excit2excit = (prev_GEE+K*(1-numpy.exp(-t/tau_NE)))
        else:
            excit_shiftcenter.G_excit2excit = (prev_GEE+K_shift*numpy.exp(-t/tau_NE))
            

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        rate_monitor = PopulationRateMonitor(pop)
        # record= some_list is not supported? :-(
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

    # collect data of a subset of neurons:
    rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
        get_monitors(inhib_pop, monitored_subset_size, N_inhibitory)

    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size, N_excitatory)
        

    b2.run(sim_time)
    return \
        excit_center.G_excit2excit, rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45

########################################Execute WM function###############################################
        
def getting_started():
    # Parameters
    b2.defaultclock.dt = 0.1 * b2.ms
    # params
    N_e=1024
    simtime = 5000. * b2.ms
    learning_rate = 0.05*b2.nS
    weight_factor = 2.      # changes inverse proportionally to site of neuronpool
    stimulus_center=180
    strength = 0.3*10**(-9)
    num_steps = 16
    distr_deg_max = 55
    
    ### Calculate strength, time course of distractor as inverted parabola ###
    distr_strength = numpy.zeros(2*int(num_steps/2)+1)
    distr_start = numpy. zeros(2*int(num_steps/2)+1)
    for k in range(int(-num_steps/2),int(num_steps/2)+1):
        distr_strength[k+int(num_steps/2)] = ((-(k)**2+ int(num_steps/2)**2)/(int(num_steps/2)**2))*strength
        distr_start[k+int(num_steps/2)] = 0+(k+int(num_steps/2))*simtime/num_steps 
    distr_strength = distr_strength*b2.amp
    distr_start = distr_start*b2.second
    #print('full strength = '+str(distr_strength))
    #print('times for increasing current'+str(distr_start))
        #distr_end[t] = (t+1)*simtime/num_steps
    
    ### Implement learning based on G_EE ###
    result_file = "/home/melanie/Schreibtisch/MSNE/NISE/results/Norepinephrin2501.txt"
    ### Define experiment set-up ###
    setmax = 1
    adaptation_len = 7
    washout_len = 2
    trialmax = adaptation_len+washout_len
    for sets in range(0,setmax):
        for trials in range(0,trialmax):
            d_strength = distr_strength
            if trials == 0:
                GEE = 0.35*0.381 * b2.nS*weight_factor
                err = stimulus_center
                adapt = True
                print('Adaptation trial')
            elif (trials < adaptation_len):
                file = get_results(result_file)
                GEE = file[(trials)-1+(sets)*trialmax, 3]*b2.nS
                err = file[(trials)-1+(sets)*trialmax, 2]  
                adapt = True
                print('Adaptation trial')
            else:
                d_strength=numpy.zeros(2*int(num_steps/2)+1)*b2.amp
                file = get_results(result_file)
                GEE = file[(trials)-1+(sets)*trialmax, 3]*b2.nS
                err = file[(trials)-1+(sets)*trialmax, 2]
                adapt = False
                print('Wash-out trial')
               
            print('Set No. '+str(sets)+', Trial No. '+str(trials))
    ### Call working memory function ###                 
            G_ee, rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
                rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
                weight_profile\
                = simulate_wm(N_excitatory=N_e, N_inhibitory=256,
                              weight_scaling_factor=weight_factor, sim_time=simtime, poisson_firing_rate=0.9*b2.Hz,
                              error = err, prev_GEE = GEE, 
                              learning_factor=learning_rate, adapt = adapt,
                              stimulus_center_deg=stimulus_center ,stimulus_width_deg=10, 
                              t_stimulus_start=0 * b2.ms, t_stimulus_duration=200 * b2.ms, stimulus_strength=0.4*b2.nA, 
                              distractor_center_deg=distr_deg_max+180, distr_steps = num_steps, distractor_width_deg=10, 
                              distractor_strength=distr_strength, t_distractor_start=distr_start, 
                              t_distractor_duration=simtime/num_steps)
        # poisson_firing_rate=2.4* b2.Hz, Jpos_excit2excit= 1.453,
        
        
#################################################### CALCULATE ERROR ##############################################################
            i,t = spike_monitor_excit.it
                
            spike_list=numpy.zeros(N_e)
            for k in range(0,N_e):
                spike_list[k] = numpy.where(i[numpy.where(numpy.logical_and(t>=simtime-100*b2.ms, t<simtime))[0]]==k)[0].size
            bump_center = decode(spike_list, N_e)
            print(bump_center*360/(2*numpy.pi))
        
        #################################################### PLOTS ########################################################################
            plt.rcParams.update({'font.size': 14})
            plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit,
                                              t_min=0. * b2.ms)
            plt.show()
            
            plt.plot(t, i, 'k.', ms=3)
            plt.plot([0, simtime], [180*N_e/(360),180*N_e/(360)])
            plt.xlabel('time [s]')
            plt.ylabel('neuron idx')
            #for k in range(0,num_steps):
            #    plt.fill_between([a[k], a[k]+simtime/num_steps],[(distr_deg[k]-0.5*5)*N_e/(360),(distr_deg[k]-0.5*5)*N_e/(360)], [(distr_deg[k]+0.5*5)*N_e/(360),(distr_deg[k]+0.5*5)*N_e/(360)])
            #for k in range(0,num_steps):
            #    plt.plot(distr_start[k], distr_deg[k]*N_e/(360), 'b.', MarkerSize=10)
            #plt.savefig('/home/melanie/Schreibtisch/MSNE/NISE/results/ExamplePerturbed.png', dpi=300)
            plt.show()
            
            # Plot the externel current strength of distractor
            #x = numpy.linspace(0,num_steps+1,num_steps+1)*simtime/num_steps
            #plt.xlabel('time [s]')
            #plt.ylabel('I_ext [A]')
            #plt.plot(x, distr_strength)
            #plt.savefig('Iext_Force2.png', dpi=300)
            #plt.show()

#################################################### SAVE TO FILE #################################################################
    
            with open(result_file, "a") as myfile:
                myfile.write(str(stimulus_center)+" "+str(distr_deg_max+180)+" "+str(bump_center*360/(2*numpy.pi))+" "+str(G_ee[int(len(G_ee)/2.)]/b2.nS)+' '+str(learning_rate/b2.nS)+' '+str(setmax)+' '+str(trialmax)+"\n")
        
            
    

if __name__ == "__main__":
    getting_started()
