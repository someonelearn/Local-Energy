# -*- coding: utf-8 -*-
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from datasets import Dataset
import io

class GaussianNoise:
  def __init__(self, mean=0, std=0, n_noised_img=3):
    self.mean = mean
    self.std = std
    self.n_noised_img = n_noised_img

  def transform_type(self, image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    png_image = Image.open(buffer)
    return png_image

  def add_noise(self, image):
    #convert a PIL Image to noisy PIL Image
    img_arr = np.array(image)

    noise = np.random.normal(self.mean, self.std, img_arr.shape).astype(np.int16)
    noised_img_arr = img_arr +  noise
    noised_img_arr = np.clip(noised_img_arr, 0, 255).astype(np.uint8)

    noised_img = Image.fromarray(noised_img_arr)
    noised_img = self.transform_type(noised_img)
    return noised_img

  def add_noise_data(self, dataset):
    gaussian_ds = []
    for c in tqdm(range(self.n_noised_img)):
            gaussian_ds.append({
                "image" : transform_type(self.add_noise(img)),
                "label" : label
            })
    return Dataset.from_list(gaussian_ds)
