# Import the required libraries
import tkinter as tk  # Import the tkinter module for GUI
import sim  # Import the sim module for CoppeliaSim interaction

# Define a class for remote control
class RemoteControl:
    ClientID = 0  # Client ID for the connection
    Joint_handles = []  # List to store joint handles

    Num_joints = 7  # Number of joints in the robot

    def Connect(self):
        try:
            # Establish a connection with CoppeliaSim
            self.ClientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        except Exception as e:
            print('Connection failed:', str(e))
            exit()

    def DisConnect(self):
        # Disconnect from the simulation
        sim.simxFinish(self.ClientID)

    def move_joint(self, index, value):
        # Move a specific joint to the given value
        joint_angle = float(value)
        joint_handle = self.Joint_handles[index]
        sim.simxSetJointTargetPosition(self.ClientID, joint_handle, joint_angle, sim.simx_opmode_oneshot)

    def get_joint_handles(self):
        # Retrieve the handles of the robot's joints
        for i in range(self.Num_joints):
            if i == 0:
                joint_name = 'joint'
            else:
                joint_name = 'LBR_iiwa_7_R800_joint' + str(i + 1)

            _, joint_handle = sim.simxGetObjectHandle(self.ClientID, joint_name, sim.simx_opmode_blocking)
            self.Joint_handles.append(joint_handle)
        return self.Joint_handles



# Create an instance of the RemoteControl class
remoteControl = RemoteControl()

def button_state_trigger(trigger):
    # Update button states based on the trigger value
    if trigger:
        connect_button.config(state="disabled")  # Disable the Connect button
        disconnect_button.config(state="normal")  # Enable the Disconnect button
        for i in range(remoteControl.Num_joints):
            scrollbars[i].config(state="normal")  # Enable the scrollbars
            scrollbars[i].set(0)  # Set scrollbar values to zero
    else:
        connect_button.config(state="normal")  # Enable the Connect button
        disconnect_button.config(state="disabled")  # Disable the Disconnect button
        for i in range(remoteControl.Num_joints):
            scrollbars[i].config(state="disabled")  # Disable the scrollbars


def connect():
    # Set up the connection with CoppeliaSim
    remoteControl.Connect()

    # Check if the connection was successful
    if remoteControl.ClientID == -1:
        print('Connection failed')
        exit()
    else:
        remoteControl.get_joint_handles()
        button_state_trigger(True)


def disconnect():
    # Disconnect from CoppeliaSim
    remoteControl.DisConnect()
    button_state_trigger(False)

# Create a Tkinter window
window = tk.Tk()
window.title("Robot Control")

# Create joint scrollbars
scrollbars = []
for i in range(remoteControl.Num_joints):
    frame = tk.Frame(window)
    frame.grid(row=0, column=i, padx=10, pady=10)

    scrollbar = tk.Scale(frame, from_=180, to=-180, orient="vertical",
                         command=lambda value, index=i: remoteControl.move_joint(index, value))
    scrollbar.pack(side="top")
    scrollbars.append(scrollbar)

    joint_label = tk.Label(frame, text="J" + str(i + 1))
    joint_label.pack(side="right")

# Create the Connect button
frame = tk.Frame(window)
frame.grid(row=1, column=0, columnspan=3)
connect_button = tk.Button(frame, text="Connect", command=connect)
connect_button.pack(side="left")

frame = tk.Frame(window)
frame.grid(row=1, column=5, columnspan=3)
# Create the Disconnect button
disconnect_button = tk.Button(frame, text="Disconnect", command=disconnect)
disconnect_button.pack(side="right")

button_state_trigger(False)

# Run the Tkinter window
window.mainloop()
