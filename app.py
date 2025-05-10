import gradio as gr
import os
import uuid
from inference import infer as run_inference  # Changed from main to infer
from PIL import Image

def generate_video(image, prompt, height, width, num_frames, seed):
    input_image_path = f"/tmp/input_{uuid.uuid4().hex}.png"
    image.save(input_image_path)

    run_inference(
        output_path=None,
        seed=seed,
        pipeline_config="configs/ltxv-13b-0.9.7-dev.yaml",
        image_cond_noise_scale=0.15,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=30,
        prompt=prompt,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        offload_to_cpu=False,
        conditioning_media_paths=[input_image_path],
        conditioning_strengths=[1.0],
        conditioning_start_frames=[0],
    )

    output_dir = "outputs"
    today_folder = sorted(os.listdir(output_dir))[-1]
    video_folder = os.path.join(output_dir, today_folder)
    video_file = sorted(
        [f for f in os.listdir(video_folder) if f.endswith(".mp4")],
        key=lambda x: os.path.getmtime(os.path.join(video_folder, x))
    )[-1]
    video_path = os.path.join(video_folder, video_file)

    return video_path

iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Prompt"),
        gr.Slider(256, 1024, value=512, step=64, label="Height"),
        gr.Slider(256, 1024, value=512, step=64, label="Width"),
        gr.Slider(8, 32, value=16, step=1, label="Number of Frames"),
        gr.Slider(0, 99999, value=42, step=1, label="Seed"),
    ],
    outputs=gr.Video(label="Generated Video"),
    title="LTX-Singh - AI Video Generator"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
