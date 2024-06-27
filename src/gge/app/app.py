import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

# Import sensors using absolute import
from gge.sensors import Sentinel1, Sentinel2, Landsat


class SatelliteImageTool:
    def __init__(self, master):
        self.master = master
        self.master.title("Satellite Image Acquisition Tool")

        # Setup Frames
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        # Date Selection
        self.start_date_label = tk.Label(self.frame, text="Start Date (YYYY-MM-DD):")
        self.start_date_label.grid(row=0, column=0)
        self.start_date_entry = tk.Entry(self.frame)
        self.start_date_entry.grid(row=0, column=1)

        self.end_date_label = tk.Label(self.frame, text="End Date (YYYY-MM-DD):")
        self.end_date_label.grid(row=1, column=0)
        self.end_date_entry = tk.Entry(self.frame)
        self.end_date_entry.grid(row=1, column=1)

        # Area Selection
        self.area_button = tk.Button(self.frame, text="Select Area on Map", command=self.select_area_on_map)
        self.area_button.grid(row=2, column=0, columnspan=2)

        # Sensor Selection
        self.sensor_label = tk.Label(self.frame, text="Select Sensor:")
        self.sensor_label.grid(row=3, column=0)
        self.sensor_var = tk.StringVar(self.frame)
        self.sensor_option = tk.OptionMenu(self.frame, self.sensor_var, "Sentinel-1", "Sentinel-2", "Landsat-8")
        self.sensor_option.grid(row=3, column=1)

        # Fetch and Display Data
        self.fetch_button = tk.Button(self.frame, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.grid(row=4, column=0, columnspan=2)

        # Save Data
        self.save_button = tk.Button(self.frame, text="Save Images as .npy", command=self.save_images_as_npy)
        self.save_button.grid(row=5, column=0, columnspan=2)

        # Placeholder for image display (matplotlib integration)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.rs = RectangleSelector(self.ax, self.line_select_callback, interactive=True, button=[1], minspanx=5, minspany=5)

        self.selected_area = None
        self.image_data = None

    def line_select_callback(self, eclick, erelease):
        "eclick and erelease are the press and release events"
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.selected_area = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        print("Selected area: (%.2f, %.2f) to (%.2f, %.2f)" % (x1, y1, x2, y2))

    def select_area_on_map(self):
        # Use matplotlib's ginput to simulate map clicks (for demonstration, replace with real map data handling)
        plt.figure()
        plt.title("Click to select an area (Close window when done)")
        plt.imshow(np.random.rand(10, 10))  # Placeholder for map
        plt.show()

    def fetch_data(self):
        # Fetch data based on selected sensor
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        area = self.selected_area
        sensor = self.sensor_var.get()

        if sensor == "Sentinel-1":
            data = Sentinel1.fetch_data(start_date, end_date, area)
        elif sensor == "Sentinel-2":
            data = Sentinel2.fetch_data(start_date, end_date, area)
        elif sensor == "Landsat-8":
            data = Landsat.fetch_data(start_date, end_date, area)
        else:
            messagebox.showerror("Error", "Unsupported sensor type.")
            return

        self.image_data = data  # Assuming data is returned as a NumPy array
        self.ax.imshow(self.image_data)
        self.canvas.draw()

    def save_images_as_npy(self):
        # Save the fetched image data as .npy file
        if self.image_data is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
            if save_path:
                np.save(save_path, self.image_data)
                messagebox.showinfo("Save", f"Image data saved as {save_path}.")
        else:
            messagebox.showwarning("Save", "No image data to save.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SatelliteImageTool(root)
    root.mainloop()
