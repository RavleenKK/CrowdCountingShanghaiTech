
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import cv2


st.set_page_config(
    page_title="Crowd Counting AI",
    page_icon="👥",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 48px;
    font-weight: 700;
    color: #111827;
}

.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 20px;
}

.metric-box {
    background: #eef2ff;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
}

.metric-box h1 {
    color: #1d4ed8;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1e40af;
    color: white;
}

</style>
""", unsafe_allow_html=True)

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        mid = max(channels // reduction, 8)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        return self.sig(
            self.fc(self.avg(x)) + self.fc(self.max(x))
        )


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            7,
            padding=3,
            bias=False
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        avg = x.mean(dim=1, keepdim=True)

        mx = x.max(dim=1, keepdim=True).values

        return self.sig(
            self.conv(torch.cat([avg, mx], dim=1))
        )


class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels)

        self.sa = SpatialAttention()

    def forward(self, x):

        return x * self.ca(x) * self.sa(x)


def dilated_block(in_ch, out_ch, d=2):

    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=d,
            dilation=d,
            bias=False
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class CrowdCountingNet(nn.Module):

    def __init__(self, pretrained=False):

        super().__init__()

        vgg = models.vgg16_bn(
            weights='DEFAULT' if pretrained else None
        )

        self.frontend = nn.Sequential(
            *list(vgg.features.children())[:33]
        )

        self.backend = nn.Sequential(

            dilated_block(512, 512, 1),

            dilated_block(512, 512, 2),

            dilated_block(512, 512, 2),

            dilated_block(512, 256, 2),

            dilated_block(256, 128, 2),

            dilated_block(128, 64, 2),
        )

        self.attention = CBAM(64)

        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.frontend(x)

        x = self.backend(x)

        x = self.attention(x)

        x = self.output_conv(x)

        return x

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CrowdCountingNet().to(device)

# IMPORTANT:
# Put best_model.pth in same folder

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()


def predict(image):

    original_h, original_w, _ = image.shape

    # Resize image
    resized = cv2.resize(image, (512, 512))

    # Convert to tensor
    tensor = torch.from_numpy(resized).float()

    tensor = tensor.permute(2, 0, 1) / 255.0

    # Normalize
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3,1,1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3,1,1)

    tensor = (tensor - mean) / std

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

    density_map = output.squeeze().cpu().numpy()

    # Resize density map to original size
    density_map = cv2.resize(
        density_map,
        (original_w, original_h)
    )

    count = output.sum().item()

    return count, density_map


st.markdown(
    "<div class='title'>👥 Crowd Counting AI System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>CSRNet + CBAM Attention based Crowd Estimation using Deep Learning</div>",
    unsafe_allow_html=True
)

st.write("")


st.sidebar.title("📌 Model Information")

st.sidebar.success("""
### Architecture
CSRNet + CBAM Attention

### Dataset
ShanghaiTech Dataset

### Features
- Crowd Estimation
- Density Map Visualization
- Deep Learning Prediction
- Real-time Inference
""")


uploaded_file = st.file_uploader(
    "Upload Crowd Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    st.write("")

    col1, col2 = st.columns(2)


    with col1:

        st.markdown("## Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )



    with st.spinner("Predicting Crowd Count..."):

        predicted_count, density_map = predict(image_np)



    with col2:

        st.markdown("## Density Map")

        fig, ax = plt.subplots(figsize=(7,7))

        ax.imshow(density_map, cmap='jet')

        ax.axis("off")

        st.pyplot(fig)

    st.write("")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.markdown(
            f"""
            <div class='metric-box'>
                <h3>Predicted Count</h3>
                <h1>{int(predicted_count)}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:

        st.markdown(
            """
            <div class='metric-box'>
                <h3>Architecture</h3>
                <h1>CSRNet</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:

        st.markdown(
            """
            <div class='metric-box'>
                <h3>Attention</h3>
                <h1>CBAM</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")


    st.success(
        f"Estimated Crowd Count: {int(predicted_count)} people"
    )

else:

    st.info("Please upload an image to begin prediction.")


st.write("---")

st.markdown(
    "Built using PyTorch, Streamlit, CSRNet and CBAM Attention"
)
