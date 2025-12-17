import streamlit as st
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim


MODEL_ID = "timbrooks/instruct-pix2pix"

st.set_page_config(page_title="Production Uniquifier", page_icon="🛡️", layout="wide")


st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0052D4, #4364F7, #6FB1FC);
        color: white; font-weight: bold; padding: 0.8rem; border-radius: 8px; border: none;
        font-size: 1.1em;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .metric-container {
        border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_local_model():
    if not torch.cuda.is_available():
        st.error("❌ GPU NVIDIA не найдена!")
        return None, "cpu"

    device = "cuda"
    dtype = torch.float16 
    
    try:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=dtype, 
            safety_checker=None
        )
        pipe.to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        return pipe, device
    except Exception as e:
        st.error(f"Ошибка модели: {e}")
        return None, "cpu"


def calculate_metrics(img1, img2):
    size = (512, 512)
    i1 = np.array(img1.convert('L').resize(size))
    i2 = np.array(img2.convert('L').resize(size))
    similarity, _ = ssim(i1, i2, full=True)
    return (1 - similarity) * 100


st.title("🛡️ Uniquifier: Production Ready")
st.caption(f"Модель: {MODEL_ID} • GPU Active")

pipe, device = load_local_model()


with st.sidebar:
    st.header("⚙️ Панель управления")
    uploaded_file = st.file_uploader("Файл изображения", type=["jpg", "png", "webp"])
    
    st.markdown("---")
    
    
    neutral_prompt = "Create a high-quality variation. Enhance details, lighting, and textures. Maintain the original style and composition."
    prompt = st.text_area("Инструкция (Промпт)", value=neutral_prompt, height=100)
    
    st.markdown("---")
    
    
    steps = st.slider("Детализация (Steps)", 15, 50, 26)
    
    
    image_guidance = st.slider(
        "Привязка к оригиналу (Image Guidance)", 
        1.0, 2.5, 1.60, 0.05, 
        help="1.60 - Жесткая привязка к оригиналу (изменения только в деталях)"
    )
    
    text_guidance = st.slider("Сила промпта (Text Guidance)", 5.0, 10.0, 7.5)


if uploaded_file and pipe:
    original = Image.open(uploaded_file).convert("RGB")    
    
    w, h = original.size    
    max_dim = 1536 
    if w > max_dim or h > max_dim:
        ratio = min(max_dim/w, max_dim/h)
        w, h = int(w*ratio), int(h*ratio)
    
 
    w, h = (w // 8) * 8, (h // 8) * 8
    original = original.resize((w, h))

    run_btn = st.button("🚀 ЗАПУСТИТЬ ПРОЦЕСС", use_container_width=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходник")
        st.image(original, use_container_width=True)

    if run_btn:
        with col2:
            st.subheader("Результат")
            
           
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Инициализация...")
            start_t = time.time()
            
            
            def progress_callback(step, timestep, latents):
                
                pct = (step + 1) / steps
                progress_bar.progress(min(pct, 1.0))
                status_text.text(f"Генерация: Шаг {step + 1} из {steps}")

            try:
                
                torch.cuda.empty_cache()
                
                result = pipe(
                    prompt, 
                    image=original, 
                    num_inference_steps=steps, 
                    image_guidance_scale=image_guidance,
                    guidance_scale=text_guidance,
                    callback=progress_callback,
                    callback_steps=1
                ).images[0]
                
                duration = time.time() - start_t
                status_text.empty()
                progress_bar.empty()
                
                st.image(result, use_container_width=True)

              
                st.markdown("### 📊 Отчет")
                diff = calculate_metrics(original, result)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Время", f"{duration:.1f} сек")
                
                if 5.0 <= diff <= 25.0:
                    status = "✅ В рамках ТЗ"
                    color = "normal"
                elif diff < 5.0:
                    status = "⚠️ Мало изменений"
                    color = "off"
                else:
                    status = "⚠️ Сильные изменения"
                    color = "inverse"

                m2.metric("Отличие (Diff)", f"{diff:.2f}%", status, delta_color=color)
                m3.metric("Схожесть (SSIM)", f"{(100-diff)/100:.3f}")

                
                from io import BytesIO
                buf = BytesIO()
                result.save(buf, format="PNG")
                st.download_button("📥 СКАЧАТЬ (PNG)", buf.getvalue(), "uniq_result.png", "image/png", use_container_width=True)
            
            except Exception as e:
                st.error(f"Ошибка генерации: {e}")
                st.error("Попробуйте уменьшить размер изображения, если ошибка связана с памятью (OOM).")

elif not uploaded_file:
    st.info("👈 Загрузите файл.")