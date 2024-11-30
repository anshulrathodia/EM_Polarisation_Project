import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# Function to calculate perpendicular polarization fields
def calculate_fields(Ei0, omega, epsilon_1, mu_1, theta_i_deg, z_min, z_max, x_min, x_max):
    # Derived quantities
    c = 3e8  # Speed of light in vacuum
    k1 = omega * np.sqrt(mu_1 * epsilon_1)  # Wave number in medium 1
    lambda_ = 2 * np.pi / k1  # Wavelength in medium 1
    theta_i = np.radians(theta_i_deg)  # Angle of incidence in radians
    beta_1 = k1
    z = np.linspace(z_min, z_max, 500)
    x = np.linspace(x_min, x_max, 100)

    # Electric field E1_y
    E1_y = -1j * 2 * Ei0 * np.sin(beta_1 * x[:, None] * np.cos(theta_i)) * \
           np.exp(-1j * beta_1 * z * np.sin(theta_i))

    # Magnetic fields Hx and Hz
    Hx = np.cos(theta_i) * np.cos(beta_1 * x[:, None] * np.cos(theta_i)) * \
         np.exp(-1j * beta_1 * z * np.sin(theta_i))
    Hz = 1j * np.sin(theta_i) * np.sin(beta_1 * x[:, None] * np.cos(theta_i)) * \
         np.exp(-1j * beta_1 * z * np.sin(theta_i))

    return z, x, E1_y, Hx, Hz, lambda_, beta_1

# Function to calculate parallel polarization fields
def calculate_parallel_fields(Ei0, omega, epsilon_1, mu_1, theta_i_deg, z_min, z_max, x_min, x_max):
    # Derived quantities
    k1 = omega * np.sqrt(mu_1 * epsilon_1)  # Wave number in medium 1
    lambda_ = 2 * np.pi / k1  # Wavelength in medium 1
    eta_1 = np.sqrt(mu_1 / epsilon_1)  # Intrinsic impedance of medium 1
    theta_i = np.radians(theta_i_deg)  # Convert angle to radians
    beta_1 = k1
    z = np.linspace(z_min, z_max, 500)  # Spatial range for z
    x = np.linspace(x_min, x_max, 100)  # Range for x

    # Electric fields
    Ez = -2 * Ei0 * np.cos(theta_i) * np.sin(beta_1 * x[:, None] * np.cos(theta_i)) * \
         np.exp(-1j * beta_1 * z * np.sin(theta_i))
    Ex = -2 * Ei0 * np.sin(theta_i) * np.cos(beta_1 * x[:, None] * np.cos(theta_i)) * \
         np.exp(-1j * beta_1 * z * np.sin(theta_i))
    E_total = Ex + Ez  # Total electric field

    # Magnetic field
    H1 = (2 * Ei0 / eta_1) * np.cos(beta_1 * x[:, None] * np.cos(theta_i)) * \
         np.exp(-1j * beta_1 * z * np.sin(theta_i))

    return z, x, Ez, Ex, E_total, H1, lambda_, beta_1

# Streamlit app
st.set_page_config(layout="wide")  # Full-width layout
st.title("Electromagnetic Wave Visualization")

# Sidebar for parameter inputs
st.sidebar.header("Parameters")
Ei0 = st.sidebar.number_input("Enter E_i0:", value=1.0)
omega = st.sidebar.number_input("Enter ω (rad/s):", value=2 * np.pi * 1e9)
epsilon_r1 = st.sidebar.number_input("Enter ε_r1 (Relative Permittivity):", value=1.0)
mu_r1 = st.sidebar.number_input("Enter μ_r1 (Relative Permeability):", value=1.0)
theta_i_deg = st.sidebar.slider("Enter θ_i (degrees):", min_value=0, max_value=90, value=45)

# Allow the user to adjust the range for x and z dynamically
z_min = st.sidebar.number_input("Enter z_min (m):", value=-2.0, step=0.1)
z_max = st.sidebar.number_input("Enter z_max (m):", value=2.0, step=0.1)
x_min = st.sidebar.number_input("Enter x_min (m):", value=-1.0, step=0.1)
x_max = st.sidebar.number_input("Enter x_max (m):", value=1.0, step=0.1)

# Toggle between perpendicular and parallel polarization
mode = st.radio("Select Polarization Mode:", ("Perpendicular", "Parallel"))

# Calculate absolute permittivity and permeability
epsilon_1 = epsilon_r1 * 8.85e-12  # Absolute permittivity in F/m
mu_1 = mu_r1 * (4 * np.pi * 1e-7)  # Absolute permeability in H/m

if mode == "Perpendicular":
    st.subheader("Perpendicular Polarization Mode")

    # Calculate fields for perpendicular polarization
    z, x, E1_y, Hx, Hz, lambda_, beta_1 = calculate_fields(Ei0, omega, epsilon_1, mu_1, theta_i_deg, z_min, z_max, x_min, x_max)

    # Plot Layout (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # Electric Field E1_y
    axs[0, 0].plot(z, np.real(E1_y[0, :]), label="E_{-y} (Real)")
    axs[0, 0].set_title("Electric Field E_{-y}(z)")
    axs[0, 0].set_xlabel("z (m)")
    axs[0, 0].set_ylabel("E_{-y} (V/m)")
    axs[0, 0].grid()
    axs[0, 0].legend()

    # Magnetic Field Hx
    axs[0, 1].plot(z, np.real(Hx[0, :]), label="H_x (Real)", color="red")
    axs[0, 1].set_title("Magnetic Field H_x(z)")
    axs[0, 1].set_xlabel("z (m)")
    axs[0, 1].set_ylabel("H_x (A/m)")
    axs[0, 1].grid()
    axs[0, 1].legend()

    # Magnetic Field Hz
    axs[1, 0].plot(z, np.real(Hz[0, :]), label="H_z (Real)", color="green")
    axs[1, 0].set_title("Magnetic Field H_z(z)")
    axs[1, 0].set_xlabel("z (m)")
    axs[1, 0].set_ylabel("H_z (A/m)")
    axs[1, 0].grid()
    axs[1, 0].legend()

    # 2D Representation of Electric Field E1_y
    Z, X = np.meshgrid(z, x)
    E1_y_2D = np.real(-1j * 2 * Ei0 * np.sin(beta_1 * X * np.cos(np.radians(theta_i_deg))) *
                      np.exp(-1j * beta_1 * Z * np.sin(np.radians(theta_i_deg))))
    pcm = axs[1, 1].pcolormesh(Z, X, E1_y_2D, cmap="viridis", shading="auto")
    axs[1, 1].set_title("2D Electric Field E_{-y}(z, x)")
    axs[1, 1].set_xlabel("z (m)")
    axs[1, 1].set_ylabel("x (m)")
    fig.colorbar(pcm, ax=axs[1, 1], label="E_{-y} (V/m)")

    st.pyplot(fig, use_container_width=True)

elif mode == "Parallel":
    st.subheader("Parallel Polarization Mode")

    # Calculate fields for parallel polarization
    z, x, Ez, Ex, E_total, H1, lambda_, beta_1 = calculate_parallel_fields(Ei0, omega, epsilon_1, mu_1, theta_i_deg, z_min, z_max, x_min, x_max)

    # Plot Layout (3x2 grid) to include H1 plot
    fig, axs = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)

    # Electric Field Ez
    axs[0, 0].plot(z, np.real(Ez[0, :]), label="E_z (Real)", color="blue")
    axs[0, 0].set_title("Electric Field E_z(z)")
    axs[0, 0].set_xlabel("z (m)")
    axs[0, 0].set_ylabel("E_z (V/m)")
    axs[0, 0].grid()
    axs[0, 0].legend()

    # Electric Field Ex
    axs[0, 1].plot(z, np.real(Ex[0, :]), label="E_x (Real)", color="red")
    axs[0, 1].set_title("Electric Field E_x(z)")
    axs[0, 1].set_xlabel("z (m)")
    axs[0, 1].set_ylabel("E_x (V/m)")
    axs[0, 1].grid()
    axs[0, 1].legend()

    # Total Electric Field E_total
    axs[1, 0].plot(z, np.real(E_total[0, :]), label="E_{total} (Real)", color="green")
    axs[1, 0].set_title("Total Electric Field E_{total}(z)")
    axs[1, 0].set_xlabel("z (m)")
    axs[1, 0].set_ylabel("E_{total} (V/m)")
    axs[1, 0].grid()
    axs[1, 0].legend()

    # 2D Representation of Electric Field E_total
    Z, X = np.meshgrid(z, x)
    E_total_2D = np.real(-2 * Ei0 * np.cos(np.radians(theta_i_deg)) *
                         np.sin(beta_1 * X * np.cos(np.radians(theta_i_deg))) *
                         np.exp(-1j * beta_1 * Z * np.sin(np.radians(theta_i_deg))))
    pcm = axs[1, 1].pcolormesh(Z, X, E_total_2D, cmap="plasma", shading="auto")
    axs[1, 1].set_title("2D Total Electric Field E_{total}(z, x)")
    axs[1, 1].set_xlabel("z (m)")
    axs[1, 1].set_ylabel("x (m)")
    fig.colorbar(pcm, ax=axs[1, 1], label="E_{total} (V/m)")

    # Plot Magnetic Field H1 (Real and Imaginary Components) in the bottom row
    axs[2, 0].plot(z, np.real(H1[0, :]), label="H_1 (Real)", color="purple")
    axs[2, 0].plot(z, np.imag(H1[0, :]), label="H_1 (Imag)", color="orange", linestyle="--")
    axs[2, 0].set_title("Magnetic Field H_1(z)")
    axs[2, 0].set_xlabel("z (m)")
    axs[2, 0].set_ylabel("H_1 (A/m)")
    axs[2, 0].grid()
    axs[2, 0].legend()

    # Empty plot for the second column of the last row (you can remove this if you prefer)
    axs[2, 1].axis('off')

    st.pyplot(fig, use_container_width=True)

