import time
import uuid

import cv2
import gradio as gr
import numpy as np
import spaces
import supervision as sv
import torch

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Detect if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model from Hugging Face
processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
model = AutoModelForZeroShotObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf").to(device)

# Custom CSS to enhance text area visibility
css = """
.feedback textarea {font-size: 24px !important}
"""

# Initialize global variables
global classes
global detections
global labels
global threshold

# Set default values
classes = "person, university, class, Liectenstein"
detections = None
labels = None
threshold = 0.2

# Instantiate annotators for bounding boxes, masks, and labels
BOX_ANNOTATOR = sv.BoxAnnotator()  # Updated from BoundingBoxAnnotator
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

# Frame subsampling factor for video processing efficiency
SUBSAMPLE = 2

def annotate_image(input_image, detections, labels) -> np.ndarray:
    """Applies mask, bounding box, and label annotations to a given image."""
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)  # Updated
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


@spaces.GPU
def process_video(input_video, confidence_threshold, classes_new, progress=gr.Progress(track_tqdm=True)):
    """Processes the input video frame by frame, performs object detection, and saves the output video."""
    global detections, labels, classes, threshold
    classes = classes_new
    threshold = confidence_threshold

    # Generate a unique file name for the output video
    result_file_name = f"output_{uuid.uuid4()}.mp4"

    # Read input video and set up output video writer
    cap = cv2.VideoCapture(input_video)
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    desired_fps = fps // SUBSAMPLE
    iterating, frame = cap.read()

    # Prepare video writer for output
    segment_file = cv2.VideoWriter(result_file_name, video_codec, desired_fps, (width, height))
    batch, frames, predict_index = [], [], []
    n_frames = 0

    while iterating:
        if n_frames % SUBSAMPLE == 0:
            predict_index.append(len(frames))
            batch.append(frame)
            frames.append(frame)

        # Process a batch of frames at once
        if len(batch) == desired_fps:
            classes_list = classes.strip().split(",")
            results, fps_value = query(batch, classes_list, threshold, (width, height))

            for i, frame in enumerate(frames):
                if i in predict_index:
                    batch_idx = predict_index.index(i)
                    detections = sv.Detections(
                        xyxy=results[batch_idx]["boxes"].cpu().detach().numpy(),
                        confidence=results[batch_idx]["scores"].cpu().detach().numpy(),
                        class_id=np.array([classes_list.index(result_class) for result_class in results[batch_idx]["classes"]]),
                        data={"class_name": results[batch_idx]["classes"]},
                    )
                    labels = results[batch_idx]["classes"]

                frame = annotate_image(input_image=frame, detections=detections, labels=labels)
                segment_file.write(frame)

            # Finalize and yield result
            segment_file.release()
            yield result_file_name, gr.Markdown(f'<h3 style="text-align: center;">Model inference FPS (batched): {fps_value * len(batch):.2f}</h3>')
            result_file_name = f"output_{uuid.uuid4()}.mp4"
            segment_file = cv2.VideoWriter(result_file_name, video_codec, desired_fps, (width, height))
            batch.clear()
            frames.clear()
            predict_index.clear()

        iterating, frame = cap.read()
        n_frames += 1


def query(frame_batch, classes, confidence_threshold, size=(640, 480)):
    """Runs inference on a batch of frames and returns the results."""
    inputs = processor(images=frame_batch, text=[classes] * len(frame_batch), return_tensors="pt").to(device)

    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        fps_value = 1 / (time.time() - start_time)

    target_sizes = torch.tensor([size[::-1]] * len(frame_batch))
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, classes=[classes] * len(frame_batch), score_threshold=confidence_threshold, target_sizes=target_sizes
    )

    return results, fps_value


def set_classes(classes_input):
    """Updates the list of classes for detection."""
    global classes
    classes = classes_input


def set_confidence_threshold(confidence_threshold_input):
    """Updates the confidence threshold for detection."""
    global threshold
    threshold = confidence_threshold_input


# Custom footer for the Gradio interface
footer = """
<div style="text-align: center; margin-top: 20px;">
    <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
    <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a> |
    <a href="https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf" target="_blank">omdet-turbo-swin-tiny-hf repo in HF</a>  
    <br>
    Made with 💖 by Pejman Ebrahimi
</div>
"""

# Gradio Interface with the customized theme and DuplicateButton
with gr.Blocks(theme='ParityError/Anime', css=css) as demo:
    gr.Markdown("## Real Time Object Detection with OmDet-Turbo")
    gr.Markdown(
        """
        This is a demo for real-time open vocabulary object detection using OmDet-Turbo.<br>
        It utilizes ZeroGPU, which allocates GPU for the first inference.<br>
        The actual inference FPS is displayed after processing, providing an accurate assessment of performance.<br>
        """
    )
    
    with gr.Row():
        input_video = gr.Video(label="Upload Video")
        output_video = gr.Video(label="Processed Video", autoplay=True)  # Removed 'streaming' argument
        actual_fps = gr.Markdown("", visible=False)
    
    with gr.Row():
        classes = gr.Textbox("person, university, class, Liectenstein", label="Objects to Detect (comma separated)", elem_classes="feedback", scale=3)
        conf = gr.Slider(label="Confidence Threshold", minimum=0.1, maximum=1.0, value=0.2, step=0.05)

    with gr.Row():
        submit = gr.Button("Run Detection", variant="primary")
        duplicate_space = gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

    example_videos = gr.Examples(
        examples=[["./UNI-LI.mp4", 0.3, "person, university, class, Liectenstein"]],
        inputs=[input_video, conf, classes],
        outputs=[output_video, actual_fps]
    )

    classes.submit(set_classes, classes)
    conf.change(set_confidence_threshold, conf)

    submit.click(
        fn=process_video,
        inputs=[input_video, conf, classes],
        outputs=[output_video, actual_fps]
    )

    gr.HTML(footer)

if __name__ == "__main__":
    demo.launch(show_error=True)