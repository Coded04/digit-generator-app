# digit-generator-app
import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Load Generator
generator = torch.load("generator.pt", map_location=torch.device('cpu'))
generator.eval()

st.title("🧠 Handwritten Digit Generator (0-9)")
digit = st.slider("Choose digit", 0, 9, 0)
generate = st.button("Generate Digit")

if generate:
    z = torch.randn(1, 100)  # latent vector
    label = torch.tensor([digit])
    with torch.no_grad():
        fake_img = generator(z, label)
    img = fake_img.squeeze().numpy()

    st.image(img, caption=f"Generated Digit: {digit}", width=200)
