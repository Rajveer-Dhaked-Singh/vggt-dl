# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
os.environ['GRADIO_DISABLE_PYI_GEN'] = '1'

import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.models.new_vggt import VGGT as NewVGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT models...")
model_old = VGGT()
model_new = NewVGGT()

_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
state_dict = torch.hub.load_state_dict_from_url(_URL)

# Load weights (strict=False for NewVGGT to allow for new temporal parameters)
model_old.load_state_dict(state_dict)
model_new.load_state_dict(state_dict, strict=False)

model_old.eval().to(device)
model_new.eval().to(device)

# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model_type="old") -> dict:
    print(f"Processing images from {target_dir} using {model_type} model")
    
    model = model_old if model_type == "old" else model_new
    model.eval()

    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)

    # Determine dtype based on hardware
    if device == "cpu":
        dtype = torch.float32
        context = torch.no_grad()
    else:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        context = torch.cuda.amp.autocast(dtype=dtype)

    with torch.no_grad():
        with context:
            predictions = model(images)

    # Post-processing
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    
    predictions['pose_enc_list'] = None 

    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    torch.cuda.empty_cache()
    return predictions

def run_both_models(target_dir):
    pred_old = run_model(target_dir, "old")
    pred_new = run_model(target_dir, "new")
    return pred_old, pred_new

# -------------------------------------------------------------------------
# 2) File Handling
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    os.makedirs(target_dir_images, exist_ok=True)
    image_paths = []

    if input_images is not None:
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) else file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) else input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps)) 
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit: break
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
            count += 1
        vs.release()

    image_paths = sorted(image_paths)
    print(f"Files ready in {target_dir_images}; took {time.time() - start_time:.3f}s")
    return target_dir, image_paths

def update_gallery_on_upload(input_video, input_images):
    if not input_video and not input_images:
        return None, None, "None", [], "Please upload data."
    target_dir, image_paths = handle_uploads(input_video, input_images)
    # Output order: reconstruction_old, reconstruction_new, target_dir_output, image_gallery, log_output
    return None, None, target_dir, image_paths, "Upload complete. Click 'Reconstruct'."

# -------------------------------------------------------------------------
# 3) Reconstruction & Visualization
# -------------------------------------------------------------------------
def gradio_demo(target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode):
    if not target_dir or target_dir == "None":
        return None, None, "Error: No data.", gr.Dropdown()

    pred_old, pred_new = run_both_models(target_dir)
    
    # Save for real-time updates
    np.savez(os.path.join(target_dir, "pred_old.npz"), **pred_old)
    np.savez(os.path.join(target_dir, "pred_new.npz"), **pred_new)

    glb_old = os.path.join(target_dir, "old.glb")
    glb_new = os.path.join(target_dir, "new.glb")

    # Generate GLBs
    for pred, path in [(pred_old, glb_old), (pred_new, glb_new)]:
        scene = predictions_to_glb(
            pred, conf_thres=conf_thres, filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg, mask_white_bg=mask_white_bg,
            show_cam=show_cam, mask_sky=mask_sky, target_dir=target_dir,
            prediction_mode=prediction_mode
        )
        scene.export(path)

    all_files = sorted(os.listdir(os.path.join(target_dir, "images")))
    choices = ["All"] + [f"{i}: {f}" for i, f in enumerate(all_files)]
    
    return glb_old, glb_new, "Reconstruction Success!", gr.Dropdown(choices=choices, value="All")

def update_visualization(target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example):
    if is_example == "True" or not target_dir or target_dir == "None":
        return None, None, "Reconstruct first to update visualization."

    # Load existing predictions
    pred_old = np.load(os.path.join(target_dir, "pred_old.npz"))
    pred_new = np.load(os.path.join(target_dir, "pred_new.npz"))

    glb_old = os.path.join(target_dir, "old_temp.glb")
    glb_new = os.path.join(target_dir, "new_temp.glb")

    results = []
    for p_data, p_path in [(pred_old, glb_old), (pred_new, glb_new)]:
        scene = predictions_to_glb(
            {k: p_data[k] for k in p_data.files}, conf_thres=conf_thres, 
            filter_by_frames=frame_filter, mask_black_bg=mask_black_bg, 
            mask_white_bg=mask_white_bg, show_cam=show_cam, mask_sky=mask_sky, 
            target_dir=target_dir, prediction_mode=prediction_mode
        )
        scene.export(p_path)
        results.append(p_path)

    return results[0], results[1], "Visualization Updated"

# -------------------------------------------------------------------------
# 4) UI Construction
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()

with gr.Blocks(theme=theme, css=".custom-log * { font-size: 20px; font-weight: bold; text-align: center; }") as demo:
    is_example = gr.Textbox(visible=False, value="None")
    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    gr.HTML("<h1>🏛️ VGGT Hybrid: Static vs. Temporal 3D</h1>")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video")
            input_images = gr.File(file_count="multiple", label="Upload Images")
            image_gallery = gr.Gallery(label="Preview", columns=4, height="300px")

        with gr.Column(scale=4):
            log_output = gr.Markdown("Upload data to begin.", elem_classes=["custom-log"])
            with gr.Row():
                reconstruction_old = gr.Model3D(label="Original VGGT (Static Mode)", height=500)
                reconstruction_new = gr.Model3D(label="New VGGT (Temporal Tracking Mode)", height=500)
            
            with gr.Row():
                submit_btn = gr.Button("Reconstruct", variant="primary")
                clear_btn = gr.Button("Clear All")

            prediction_mode = gr.Radio(["Depthmap and Camera Branch", "Pointmap Branch"], 
                                      label="Mode", value="Depthmap and Camera Branch")
            
            with gr.Row():
                conf_thres = gr.Slider(0, 100, 50, label="Confidence %")
                frame_filter = gr.Dropdown(["All"], value="All", label="Frame Filter")
            
            with gr.Row():
                show_cam = gr.Checkbox(label="Show Cam", value=True)
                mask_sky = gr.Checkbox(label="Filter Sky")
                mask_black_bg = gr.Checkbox(label="Filter Black")
                mask_white_bg = gr.Checkbox(label="Filter White")

    # --- Event Logic ---
    input_video.change(update_gallery_on_upload, [input_video, input_images], 
                      [reconstruction_old, reconstruction_new, target_dir_output, image_gallery, log_output])
    
    input_images.change(update_gallery_on_upload, [input_video, input_images], 
                       [reconstruction_old, reconstruction_new, target_dir_output, image_gallery, log_output])

    submit_btn.click(lambda: (None, None, "Processing..."), None, [reconstruction_old, reconstruction_new, log_output]).then(
        gradio_demo, 
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_old, reconstruction_new, log_output, frame_filter]
    ).then(lambda: "False", None, is_example)

    # Real-time sliders
    viz_inputs = [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example]
    for ctrl in [conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode]:
        ctrl.change(update_visualization, viz_inputs, [reconstruction_old, reconstruction_new, log_output])

    clear_btn.click(lambda: (None, None, None, [], "None", "Cleared"), None, 
                   [reconstruction_old, reconstruction_new, input_video, image_gallery, target_dir_output, log_output])

demo.launch(share=True)