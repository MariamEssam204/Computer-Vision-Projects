## CV for autonomous vechile

This project uses a fine-tuned **YOLO Nano** model to enable an underwater ROV to detect garbage in real time. The model was trained on a custom dataset of **897 training**, **80 validation**, and **40 test** images, enhanced through data augmentation (brightness/exposure shifts, color/hue adjustments, noise addition, rotation, and scaling) to simulate real underwater conditions such as variable lighting, color distortion, and blur.

Since the ROV's onboard embedded system lacks the processing power for inference, detection runs on a connected laptop via a low-latency wired link, with results sent back to guide the ROV's actions. YOLO Nano was chosen specifically for its lightweight footprint, balancing real-time performance with the available hardware constraints.
