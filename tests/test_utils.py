from PIL import Image
import numpy as np
from app import calculate_metrics 

def test_calculate_metrics_identical():
    img = Image.new("RGB", (512, 512), color="red")
    diff = calculate_metrics(img, img)
    assert 0 <= diff <= 0.1  

def test_calculate_metrics_different():
    img1 = Image.new("RGB", (512, 512), color="red")
    img2 = Image.new("RGB", (512, 512), color="blue")
    diff = calculate_metrics(img1, img2)
    assert 20 <= diff <= 100  