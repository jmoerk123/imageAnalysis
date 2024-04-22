import customtkinter as ctk
import os
from PIL import Image
import numpy as np
from imageAnalysis.src.detector import Detector, Viz


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.detector = Detector("KS")

        self.title("image_example.py")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width-500}x{screen_height-500}")
        # self.geometry("1000x550")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.image_box = (500, 450)

        ################# Navigation Frame ###############
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(3, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(
            self.navigation_frame,
            text="  Image Example",
            # image=self.logo_image,
            compound="left",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.start_frame_button = ctk.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Start",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            # image=self.home_image,
            anchor="w",
            command=self.start_frame_button_event,
        )
        self.start_frame_button.grid(row=1, column=0, sticky="ew")

        self.image_frame_button = ctk.CTkButton(
            self.navigation_frame,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Image",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            # image=self.chat_image,
            anchor="w",
            command=self.image_frame_button_event,
        )
        self.image_frame_button.grid(row=2, column=0, sticky="ew")

        self.appearance_mode_menu = ctk.CTkOptionMenu(
            self.navigation_frame,
            values=["System", "Light", "Dark"],
            command=self.change_appearance_mode_event,
        )
        self.appearance_mode_menu.grid(row=4, column=0, padx=20, pady=20, sticky="s")

        ################ Start Frame ##############
        # create home frame
        self.start_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.start_frame.grid_columnconfigure(1, weight=1)

        # Image path lapel
        self.image_path_label = ctk.CTkLabel(self.start_frame, text="Image path")
        self.image_path_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # Image path Entry Field
        self.image_path = ctk.CTkEntry(
            self.start_frame, placeholder_text="path/to/image"
        )
        self.image_path.grid(
            row=0, column=1, columnspan=3, padx=20, pady=20, sticky="ew"
        )

        self.load_image_button = ctk.CTkButton(
            self.start_frame,
            text="Load Image",
            # image=self.image_icon_image,
            command=self.load_image,
        )
        self.load_image_button.grid(row=1, column=0, padx=20, pady=10)

        # Compare-image path lapel
        self.compare_image_path_label = ctk.CTkLabel(
            self.start_frame, text="Compare-image path"
        )
        self.compare_image_path_label.grid(
            row=2, column=0, padx=20, pady=20, sticky="ew"
        )

        # compare-image path Entry Field
        self.compare_image_path = ctk.CTkEntry(
            self.start_frame, placeholder_text="path/to/compare-image"
        )
        self.compare_image_path.grid(
            row=2, column=1, columnspan=3, padx=20, pady=20, sticky="ew"
        )

        self.load_compare_image_button = ctk.CTkButton(
            self.start_frame,
            text="Load Compare Image",
            # image=self.image_icon_image,
            command=self.load_compare_image,
        )
        self.load_compare_image_button.grid(row=3, column=0, padx=20, pady=10)

        ############## Image Frame ###############
        # create second frame
        self.image_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")

        self.image = Image.open(
            "/home/jam/private/testTkinter/test_images/Sheep-scaled.jpg"
        )
        image_width, image_height = self.image.size
        image_scale = self.get_image_scale(
            image_size=(image_width, image_height), image_box=self.image_box
        )
        img_show_size = (image_width * image_scale, image_height * image_scale)
        self.ctk_image = ctk.CTkImage(
            self.image,
            size=img_show_size,
        )

        self.image_label = ctk.CTkLabel(self.image_frame, text="", image=self.ctk_image)
        self.image_label.grid(row=0, column=0, columnspan=4, padx=20, pady=10)

        self.add_keys_to_image_button = ctk.CTkButton(
            self.image_frame,
            text="Add keys",
            # image=self.image_icon_image,
            command=self.add_keys,
        )
        self.add_keys_to_image_button.grid(row=1, column=0, padx=20, pady=10)

        self.add_keys_to_image_button = ctk.CTkButton(
            self.image_frame,
            text="Add keys",
            # image=self.image_icon_image,
            command=self.add_keys,
        )
        self.add_keys_to_image_button.grid(row=1, column=0, padx=20, pady=10)

        self.add_keys_to_image_button = ctk.CTkButton(
            self.image_frame,
            text="Add keys",
            # image=self.image_icon_image,
            command=self.add_keys,
        )
        self.add_keys_to_image_button.grid(row=1, column=0, padx=20, pady=10)

        self.add_keys_from_other_image_to_image_button = ctk.CTkButton(
            self.image_frame,
            text="Add keys from other image",
            # image=self.image_icon_image,
            command=self.add_keys_from_other_image,
            state="disabled",
        )
        self.add_keys_from_other_image_to_image_button.grid(
            row=2, column=0, padx=20, pady=10
        )

        self.add_keys_from_predefined_image_to_image_button = ctk.CTkButton(
            self.image_frame,
            text="Add keys from predefined",
            # image=self.image_icon_image,
            command=self.add_keys,
        )
        self.add_keys_from_predefined_image_to_image_button.grid(
            row=3, column=0, padx=20, pady=10
        )

        ################ Default setup ##############
        self.change_appearance_mode_event("system")
        self.select_frame_by_name("start")

    def add_keys_from_other_image(self):
        self.viz.add_keypoints_from_other_img(
            image=np.array(self.compare_image)[:, :, ::-1]
        )
        self.image = Image.fromarray(self.viz.get_image())
        self.insert_image()

    def add_keys(self):
        print("add key")
        self.viz.add_keypoints()
        self.image = Image.fromarray(self.viz.get_image())
        self.insert_image()
        # viz.add_keypoints_from_other_img("images/658934.png")
        # viz.save_image("test.png")
        # viz.show_image("Results")

    def load_image(self):
        self.image = Image.open(self.image_path.get())
        self.viz = Viz(detector=self.detector, image=np.array(self.image)[:, :, ::-1])
        self.insert_image()

    def load_compare_image(self):
        self.compare_image = Image.open(self.compare_image_path.get())
        self.insert_image()
        self.add_keys_from_other_image_to_image_button.configure(state="normal")

    def insert_image(self):
        image_width, image_height = self.image.size
        image_scale = self.get_image_scale(
            image_size=(image_width, image_height), image_box=self.image_box
        )
        img_show_size = (image_width * image_scale, image_height * image_scale)
        self.ctk_image = ctk.CTkImage(
            self.image,
            size=img_show_size,
        )
        self.image_label.configure(image=self.ctk_image)

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.start_frame.configure(
            fg_color=("gray75", "gray25") if name == "home" else "transparent"
        )
        self.image_frame.configure(
            fg_color=("gray75", "gray25") if name == "frame_2" else "transparent"
        )

        # show selected frame
        if name == "start":
            self.start_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.start_frame.grid_forget()
        if name == "image":
            self.image_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.image_frame.grid_forget()

    def start_frame_button_event(self):
        self.select_frame_by_name("start")

    def image_frame_button_event(self):
        self.select_frame_by_name("image")

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

    def get_image_scale(self, image_size, image_box):
        w_scale = image_box[0] / image_size[0]
        h_scale = image_box[1] / image_size[1]
        if w_scale < 1 or h_scale < 1:
            return min([w_scale, h_scale])
        else:
            return max([w_scale, h_scale])


if __name__ == "__main__":
    app = App()
    app.mainloop()
