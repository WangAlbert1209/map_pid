# inverted_pendulum_pid.py
# Author: Nikhil Advani (Modified by ChatGPT)
# Date: 22nd May 2017 (Modified: 2024)
# Description: Simulation of a cart-pole system using PID control to balance the pendulum.

import numpy as np
import cv2
import math
import time
import matplotlib.pyplot as plt
import sys

class Cart:
    def __init__(self, x, mass, world_size):
        self.x = x  
        self.y = int(0.6 * world_size)  # 0.6 was chosen for aesthetic reasons.
        self.mass = mass
        self.color = (0, 255, 0)

class Pendulum:
    def __init__(self, length, theta, ball_mass):
        self.length = length
        self.theta = theta
        self.ball_mass = ball_mass		
        self.color = (0, 0, 255)

def display_stuff(world_size, cart, pendulum):
    """
    Displays the pendulum and cart using OpenCV.
    """
    length_for_display = pendulum.length * 100
    A = np.zeros((world_size, world_size, 3), np.uint8)
    # Draw ground
    cv2.line(A, (0, int(0.6 * world_size)), (world_size, int(0.6 * world_size)), (255, 255, 255), 2)
    # Draw cart
    cv2.rectangle(A, 
                 (int(cart.x) + 25, cart.y + 15), 
                 (int(cart.x) - 25, cart.y - 15), 
                 cart.color, -1)	
    # Calculate pendulum end position
    pendulum_x_endpoint = int(cart.x - (length_for_display) * math.sin(pendulum.theta))
    pendulum_y_endpoint = int(cart.y - (length_for_display) * math.cos(pendulum.theta))
    # Draw pendulum
    cv2.line(A, (int(cart.x), cart.y), (pendulum_x_endpoint, pendulum_y_endpoint), pendulum.color, 4)
    # Draw pendulum bob
    cv2.circle(A, (pendulum_x_endpoint, pendulum_y_endpoint), 6, (255, 255, 255), -1)
    cv2.imshow('Inverted Pendulum - PID Control', A)
    cv2.waitKey(1)  # Reduced delay for smoother simulation

def find_pid_control_input(time_delta, error, previous_error, integral):
    """
    Calculates the control force using PID controller.
    """
    # PID gains (empirically tuned)
    Kp = -250
    Kd = -30
    Ki = -230

    derivative = (error - previous_error) / time_delta
    integral += error * time_delta
    F = (Kp * error) + (Kd * derivative) + (Ki * integral)
    return F, integral

def apply_control_input(cart, pendulum, F, time_delta, x_tminus2, theta_dot, theta_tminus2, previous_time_delta, g):
    """
    Updates the state of the cart and pendulum based on the control input and system dynamics.
    """
    # Compute angular acceleration (theta_double_dot)
    numerator_theta = ((cart.mass + pendulum.ball_mass) * g * math.sin(pendulum.theta)) + \
                      (F * math.cos(pendulum.theta)) - \
                      (pendulum.ball_mass * (theta_dot**2) * pendulum.length * math.sin(pendulum.theta) * math.cos(pendulum.theta))
    denominator_theta = pendulum.length * (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2)))
    theta_double_dot = numerator_theta / denominator_theta

    # Compute linear acceleration (x_double_dot)
    numerator_x = (pendulum.ball_mass * g * math.sin(pendulum.theta) * math.cos(pendulum.theta)) - \
                  (pendulum.ball_mass * pendulum.length * math.sin(pendulum.theta) * (theta_dot**2)) + F
    denominator_x = cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2))
    x_double_dot = numerator_x / denominator_x

    # Update positions using simple Euler integration
    x_dot = (cart.x - x_tminus2) / previous_time_delta
    cart.x += x_dot * time_delta + 0.5 * x_double_dot * (time_delta**2)
    pendulum.theta += theta_dot * time_delta + 0.5 * theta_double_dot * (time_delta**2)

def find_error(pendulum):
    """
    Computes the error in pendulum angle with respect to the upright position.
    Handles angle wrapping to ensure minimal error.
    """
    error = pendulum.theta % (2 * math.pi)
    if error > math.pi:
        error -= 2 * math.pi
    return error

def plot_graphs(times, errors, theta, force, x):
    """
    Plots the simulation data: error, theta, force, and cart position over time.
    """
    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(times, errors, '-b')
    plt.ylabel('Error (rad)')
    plt.xlabel('Time (s)')
    plt.title('Pendulum Angle Error Over Time')

    plt.subplot(4, 1, 2)
    plt.plot(times, theta, '-r')
    plt.ylabel('Theta (rad)')
    plt.xlabel('Time (s)')
    plt.title('Pendulum Angle (Theta) Over Time')

    plt.subplot(4, 1, 3)
    plt.plot(times, force, '-g')
    plt.ylabel('Force (N)')
    plt.xlabel('Time (s)')
    plt.title('Control Force Over Time')

    plt.subplot(4, 1, 4)
    plt.plot(times, x, '-m')
    plt.ylabel('Position X')
    plt.xlabel('Time (s)')
    plt.title('Cart Position Over Time')

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the cart-pole simulation using PID control.
    """
    # Initialize system parameters
    mass_of_ball = 1.0
    mass_of_cart = 5.0
    g = 9.81
    world_size = 1000
    simulation_time = 35  # seconds

    # Initialize data storage for plotting
    errors, force, theta, times, x = [], [], [], [], []

    # Initialize cart and pendulum
    cart = Cart(int(0.2 * world_size), mass_of_cart, world_size)
    pendulum = Pendulum(length=1.0, theta=math.pi, ball_mass=mass_of_ball)  # Start with pendulum upright

    # Initialize simulation variables
    theta_dot = 0.0
    x_dot = 0.0
    theta_tminus1 = theta_tminus2 = pendulum.theta
    x_tminus1 = x_tminus2 = cart.x
    previous_error = find_error(pendulum)
    integral = 0.0
    previous_time_delta = 0.01  # Initialize with a small time delta to avoid division by zero
    previous_timestamp = time.time()
    end_time = previous_timestamp + simulation_time

    # Simulation loop
    while time.time() <= end_time:
        current_timestamp = time.time()
        time_delta = current_timestamp - previous_timestamp
        if time_delta <= 0:
            time_delta = 1e-5  # Prevent division by zero or negative time_delta

        # Compute error
        error = find_error(pendulum)

        # Compute derivatives only if time_delta is significant
        if time_delta > 0:
            theta_dot = (pendulum.theta - theta_tminus2) / previous_time_delta
            x_dot = (cart.x - x_tminus2) / previous_time_delta

            # Calculate control force using PID controller
            F, integral = find_pid_control_input(time_delta, error, previous_error, integral)

            # Apply control force to update system state
            apply_control_input(cart, pendulum, F, time_delta, x_tminus2, theta_dot, theta_tminus2, previous_time_delta, g)

            # Store data for plotting
            force.append(F)
            x.append(cart.x)
            errors.append(error)		
            times.append(current_timestamp - previous_timestamp)

            theta.append(pendulum.theta)

        # Display the current state
        display_stuff(world_size, cart, pendulum)

        # Update variables for next iteration
        previous_time_delta = time_delta
        previous_timestamp = current_timestamp
        previous_error = error
        theta_tminus2 = theta_tminus1
        theta_tminus1 = pendulum.theta
        x_tminus2 = x_tminus1
        x_tminus1 = cart.x

    # Close the OpenCV window
    cv2.destroyAllWindows()

    # # Plot the simulation results
    # plot_graphs(times, errors, theta, force, x)

if __name__ == "__main__":
    main()
