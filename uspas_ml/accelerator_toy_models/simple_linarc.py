import torch
import scipy.constants as cst

def propagate(bunch_n_particles=1000, bunch_rms_z=1.e-3,
        bunch_mean_E=100e6, bunch_rms_E=0.01e6,
        linac_final_E=1000e6, linac_rf_frequency=1.3e9, linac_phase=0.,
        arc_r56=0, arc_r566=0 ):
        """
        Compute the propagation of the bunch through the linac + arc

        Parameters:
        -----------
        bunch_n_particles: int
            Number of particles in the bunch

        bunch_rms_z: float (in meters)
            The RMS size of the bunch along z (before entering the linac)

        bunch_mean_E: float (in eV)
            The mean energy of the bunch (before entering the linac)

        bunch_rms_E: float (in eV)
            The RMS energy spread of the bunch (before entering the linac)

        linac_final_E: float (in eV)
            The (mean) energy of the bunch at the end of the linac

        linac_rf_frequency: float (in Hz)
            The frequency of the RF cavity of the linac

        linac_phase: float (in degrees)
            The phase of the bunch in the linac

        arc_r56, arc_r566: floats (in meter)
            The coeficients of the energy-dependent delay induced by the arc:
            z -> z + r56*delta + t566*delta**2

        Returns
        -------
        rms_z : float (meters)
            Longitudinal bunch length

        rms_delta : float (None)
            RMS bunch energy deviation from reference

        """
        # Generate the bunch before the linac, with random Gaussian distribution
        bunch_z = torch.randn(bunch_n_particles) * bunch_rms_z
        bunch_delta = torch.randn(bunch_n_particles) * bunch_rms_E / bunch_mean_E

        # Analytical change in relative energy spread (delta) after the bunch propagates in the linac
        # $\delta \rightarrow \delta \frac{E_0}{E_1} + (1- \frac{E_0}{E_1})\frac{\cos(kz +\phi)}{\cos(\phi)}$
        k = 2*cst.pi*linac_rf_frequency/cst.c
        phi = linac_phase * 2*cst.pi/360. # Convert from degrees to radians
        E0_over_E1 = bunch_mean_E/linac_final_E
        bunch_delta = E0_over_E1 * bunch_delta.clone() + \
          (1. - E0_over_E1)*(torch.cos(k*bunch_z + phi)/torch.cos(phi) -1)

        # Analytical change in position (z) after the bunch propagates in the arc
        # z -> z + r56*delta + t566*delta**2
        bunch_z = bunch_z + arc_r56*bunch_delta + \
                        arc_r566*bunch_delta**2

        #add noise to the observations
        bunch_delta += torch.randn(1)*1e-4
        bunch_z += torch.randn(1)*1e-3

        return torch.hstack((bunch_z.std(), bunch_delta.std())).reshape(1,-1)
