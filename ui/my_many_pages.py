import threading
import tkinter as tk
from tkinter import ttk, filedialog, NW
from tkinter.messagebox import showinfo, showerror

import cv2
from PIL import Image, ImageTk

from model.siamese_model import SiameseModel


class Application(tk.Tk):

    def __init__(self):
        super().__init__()

        container = tk.Frame(self)
        # container.pack(side="top", fill="both", expand=True)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (TrainFrame, PredictFrame):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("TrainFrame")

    def show_frame(self, frame_name):
        frame = self.frames[frame_name]
        frame.tkraise()


class PredictFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.filename1 = ""
        self.image1 = None
        self.canvas1 = tk.Canvas(self, bg="white", width=250, height=250)
        self.button_load_image1 = ttk.Button(self, text='Выбрать изображение', command=self.load_image1)

        opts = {'sticky': 'nswe'}
        self.canvas1.grid(row=0, column=0, **opts)
        self.button_load_image1.grid(row=1, column=0, **opts)

        self.filename2 = ""
        self.image2 = None
        self.canvas2 = tk.Canvas(self, bg="white", width=250, height=250)
        self.button_load_image2 = ttk.Button(self, text='Выбрать изображение', command=self.load_image2)

        self.canvas2.grid(row=0, column=1, **opts)
        self.button_load_image2.grid(row=1, column=1, **opts)

        self.button_predict = ttk.Button(self, text="Сравнить подписи", command=self.make_prediction)
        self.button_predict.grid(row=2, column=0, columnspan=3, **opts)

        self.predict_result_label = tk.Label(self, font=("Segoe UI", 14))
        self.predict_result_label.grid(row=3, column=0, columnspan=3, **opts)

        button = ttk.Button(self, text="Тренировать сеть", command=lambda: controller.show_frame("TrainFrame"))
        button.grid(row=4, column=0, pady=10)

        self.siamese_model = SiameseModel.load_best_model()

    @staticmethod
    def __transform_image(filename):
        image_bgr = cv2.imread(filename)
        new_size = (250, 250)
        image_bgr_resize = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr_resize, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        return ImageTk.PhotoImage(image_pil)

    def load_image1(self):
        filename = filedialog.askopenfilename(title="Выбор изображения подписи",
                                              filetypes=[("Image files", ".png .jpg .jpeg"),
                                                         ("PNG", ".png"), ("JPG", ".jpg"), ("JPEG", ".jpeg")])
        if filename:
            self.filename1 = filename
            self.image1 = self.__transform_image(filename)
            self.canvas1.create_image(0, 0, anchor=NW, image=self.image1)

    def load_image2(self):
        filename = filedialog.askopenfilename(title="Выбор изображения подписи",
                                              filetypes=[("Image files", ".png .jpg .jpeg"),
                                                         ("PNG", ".png"), ("JPG", ".jpg"), ("JPEG", ".jpeg")])
        if filename:
            self.filename2 = filename
            self.image2 = self.__transform_image(filename)
            self.canvas2.create_image(0, 0, anchor=NW, image=self.image2)

    def make_prediction(self):
        if self.image1 is not None and self.image2 is not None:
            cos_sim, confidence, predicted_label = SiameseModel.predict(self.siamese_model,
                                                                        self.filename1,
                                                                        self.filename2)
            showinfo(title="Результат проверки", message=predicted_label)
            self.predict_result_label.configure(text=predicted_label)

        else:
            showerror(title="Ошибка", message="Нужно выбрать оба изображения")


class TrainFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.label_epochs_count = ttk.Label(self, font=("Segoe UI", 14), text="Число эпох")
        validate_uint_cmd = (self.register(TrainFrame.__check_is_digit))
        self.entry_epochs_count = ttk.Entry(self, validate='all', validatecommand=(validate_uint_cmd, '%P'))
        self.entry_epochs_count.insert(0, '5')
        self.label_epochs_count.grid(row=0, column=0)
        self.entry_epochs_count.grid(row=0, column=1)

        self.label_learning_rate = ttk.Label(self, font=("Segoe UI", 14), text="Коэффициент обучения")
        validate_float_cmd = (self.register(TrainFrame.__validate_float))
        self.entry_learning_rate = ttk.Entry(self, validate='all', validatecommand=(validate_float_cmd, '%P'))
        self.entry_learning_rate.insert(0, '0.001')
        self.label_learning_rate.grid(row=1, column=0)
        self.entry_learning_rate.grid(row=1, column=1)

        self.label_batch_size = ttk.Label(self, font=("Segoe UI", 14), text="Размер мини-батча")
        self.entry_batch_size = ttk.Entry(self, validate='all', validatecommand=(validate_uint_cmd, '%P'))
        self.entry_batch_size.insert(0, '32')
        self.label_batch_size.grid(row=2, column=0)
        self.entry_batch_size.grid(row=2, column=1)

        self.button_train = ttk.Button(self, text="Тренировать модель", command=self.start_train)
        self.label_result = ttk.Label(self, font=("Segoe UI", 14), text="с.к.о.")
        self.entry_result = ttk.Entry(self, font=("Segoe UI", 14), state=tk.DISABLED)

        self.button_train.grid(row=3, column=0)
        self.label_result.grid(row=4, column=0)
        self.entry_result.grid(row=4, column=1)

        button = ttk.Button(self, text="Проверить обученную сеть",
                            command=lambda: controller.show_frame("PredictFrame"))
        button.grid(row=5, column=0, pady=10)

    @staticmethod
    def __check_is_digit(P):
        return str.isdigit(P) or P == ""

    @staticmethod
    def __validate_float(new_text):
        try:
            float(new_text)
            return True
        except ValueError:
            return False

    def __get_learning_rate(self):
        lr = float(self.entry_learning_rate.get())
        if lr <= 0 or lr >= 1:
            showerror(title="Ошибка", message="Коэффициент обучения принимает значения в интервале (0; 1)")
            return -1
        return lr

    def __get_batch_size(self):
        batch_size = int(self.entry_batch_size.get())
        if batch_size < 1:
            showerror(title="Ошибка", message="Размер мини-батча должен быть целым числом больше 0")
            return -1
        return batch_size

    def __get_epochs_count(self):
        epochs = int(self.entry_epochs_count.get())
        if epochs < 1:
            showerror(title="Ошибка", message="Число эпох должен быть целым числом больше 0")
            return -1
        return epochs

    def start_train(self):
        self.button_train.configure(state=tk.DISABLED)
        thread = threading.Thread(target=self.train_model)
        thread.start()
        self.check_thread(thread)

    def check_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.check_thread(thread))
        else:
            self.button_train.config(state=tk.NORMAL)

    def train_model(self):
        batch_size = TrainFrame.__get_batch_size(self)
        if batch_size == -1:
            return

        lr = TrainFrame.__get_learning_rate(self)
        if lr == -1:
            return

        epochs = TrainFrame.__get_epochs_count(self)
        if epochs == -1:
            return

        std_losses = SiameseModel.train_model_by_params(batch_size=batch_size, lr=lr, epochs_count=epochs)
        last_loss = std_losses[-1]
        self.entry_result.delete(0, tk.END)
        self.entry_result.insert(0, str(last_loss))


if __name__ == "__main__":
    app = Application()
    icon = tk.PhotoImage(file="../images/signature-icon-50.png")
    app.iconphoto(False, icon)
    app.title("Проверка подлинности подписи")
    app.resizable(False, False)

    app.eval('tk::PlaceWindow . center')
    app.mainloop()
