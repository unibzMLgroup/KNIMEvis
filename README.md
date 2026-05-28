# knimeVizLab

![alt text](LogoProject.png)

## Description
This repository serves as the official [UNIBZ](https://www.unibz.it) project for creating custom KNIME nodes dedicated to image processing using Python. The goal is to provide seamless integration between KNIME's workflow analytics platform and Python's advanced capabilities for image analysis and processing.

With this project, users can:

- Develop custom KNIME nodes tailored to specific image processing requirements.
- Leverage Python libraries and tools for advanced image manipulation and analysis.
- Simplify workflows by combining KNIME's no-code/low-code environment with Python's scripting flexibility.


## Installation

To start developing KNIME nodes with Python, follow these steps:
1. **Clone this repository**:
    ```bash
    git clone ...
    ```

2. **Install the KNIME Analytics Platform**
    Install [KNIME Analytics Platform](https://docs.knime.com/ap/latest/analytics_platform_installation_guide/#install-knime-analytics-platform) version 5.5.0 or higher.
    Follow the offical guide line on how to develope pure [python nod in KNIME](https://docs.knime.com/developers/latest/create_a_node_with_python/)



## Usage
Once you have completed the installation, you can begin developing and using custom KNIME nodes for image processing. To integrate your Python-based node into KNIME, follow these steps:

1. Create your custom node in Python.
2. Integrate the node into your KNIME workflow by using the custom node created in the previous step.
3. Execute the workflow, and the image processing will be handled by Python scripts through KNIME's interface.

## Available Nodes and Use Cases
### Image Reader (`ImageReader`)
* **How it works:** The node reads a column of file paths and safely loads the files into `PIL Image` objects.
* **Key Parameters:**
    * **Image Path Column:** Select the string column that contains the absolute or relative local file paths to your images.
* **Input:** A KNIME table containing at least one string column representing image locations.
* **Output:** The original table appended with a new `"Image"` column containing the loaded image objects, formatted correctly for downstream Python nodes.
### Image Denoising (`Denoising`)
* **How it works:** Applies mathematical spatial filters to pixel neighborhoods. It utilizes parallel processing (up to 16 concurrent threads) to ensure lightning-fast execution across large datasets.
* **Key Parameters:**
    * **Algorithm Selection:** 
	   *Median Filtering:* Best for removing "salt-and-pepper" noise while perfectly preserving sharp edges.
     *Gaussian Filtering:* Best for smoothing out general camera/sensor grain through a weighted blur.
    * **Filter Size:** An **odd** integer (3 to 15) defining the kernel size. Higher values remove more noise but introduce more blurring.
* **Input & Output:** Takes a KNIME table with an image column and appends a `"Denoised Image"` column. 
    *Note:* Input images are automatically converted to **8-bit Grayscale** (`L` mode) during filtering.
* **Use Cases:** Cleaning up grainy medical scans (X-rays, ultrasounds), low-light photography, or scanned historical documents before applying AI analysis or OCR.

### Edge Detection (`EdgeDetection`)
* **How it works:** Applies mathematical convolution kernels to calculate gradient magnitudes or derivatives. It utilizes parallel processing (up to 16 concurrent threads) to ensure fast execution across large image datasets.
* **Key Parameters:**
    * **Algorithm Selection:**
        * *Sobel (Standard / OpenCV):* Highlights areas of high intensity change using 3x3 kernels.
        * *Laplace:* Calculates second-order derivatives to detect rapid intensity changes (regions of local maxima/minima).
        * *Robert:* Uses diagonal 2x2 kernels. It is highly efficient for real-time applications but can be more sensitive to noise.
    * **Threshold (3-250):** A cutoff value to separate edges from the background. Gradients above this value are marked as distinct edges (white), while those below are ignored as background (black).
* **Input & Output:** Takes a KNIME table with an image column and appends an `"EdgeDetectedImage"` column.
   *Note:* Input images are automatically converted to **8-bit Grayscale** (`L` mode) during processing. The final output is a **Binary** (solid black and white) image driven by the selected threshold.
* **Use Cases:** Extracting object silhouettes, analyzing structural properties of items on an assembly line, and serving as a precursor step for contour tracing or shape recognition pipelines.

### Histogram Equalization (`Equalization`)
* **How it works:** The node calculates the input image's pixel histogram and its Cumulative Distribution Function (CDF). It then uses this CDF as a mathematical transfer function to map original pixel values to a new, wider dynamic range. The algorithm is heavily optimized using NumPy for rapid execution.
* **Key Parameters:**
    * **Image Column:** Select the column containing the images you wish to equalize. 
* **Input:** A KNIME table containing an image column. (Images are processed in 8-bit Grayscale).
 * **Output 1 (Images):** The original table appended with an `"Equalized Image"` column showing the contrast-enhanced results.
 * **Output 2 (Analytics):** A detailed statistical data table containing the numerical arrays for the `"Original Histogram"`, `"Histogram Equalized"`, and the `"Transfer Function"`.
* **Use Cases:** Enhancing under-exposed or washed-out photographs, improving the contrast of medical scans (like X-rays or MRIs) for better visibility, and standardizing lighting variances in datasets before passing them to Machine Learning pipelines.

### YOLO Object Detection & Segmentation (`YOLO`)
* **How it works:** The node processes input images using pre-trained neural networks optimized for speed and accuracy. It automatically downloads the required model weights if they are not present locally. Inference can be executed on the CPU or accelerated via a CUDA-enabled GPU.
* **Key Parameters:**
    * **Image Column:** Select the column containing the images to be analyzed.
    * **Image ID:** Select a column to act as a unique identifier to track results back to the original image.
    * **Pretrained Model:** Choose the model size based on your hardware capabilities:
        * *Nano / Small:* Fastest inference, ideal for real-time edge devices or standard CPUs.
        * *Medium / Large / Extra Large:* Highest accuracy, requires significant RAM or a dedicated GPU.
    * **Custom Model Path:** Allows users to override the default models by providing a local `.pt` file containing custom fine-tuned YOLO weights.
    * **Computation Device:** Toggle between `CPU` and `GPU` (if CUDA is installed and available).
* **Input:** A KNIME table containing a column of images and an ID column.
 * **Output 1 (Bounding Boxes):** A table listing the coordinates (`x_center`, `y_center`, `width`, `height`), `class` name, and `confidence` score for every detected object.
 * **Output 2 (Segmentation Masks):** A table providing the isolated black-and-white `masks` (as Image objects) alongside the `class` and `confidence` score.
 * **Output 3 (Image Masks):** The original input table appended with an `"ImageMasked"` column, showing the original image overlaid with colorful bounding boxes, segmentation masks, and labels for easy visual verification.
* **Use Cases:** Analyzing traffic camera footage to count vehicles, identifying defects on manufacturing lines, tracking multiple people in security footage, or processing agricultural drone images to count crops.

### Automatic Segmentation (SAM)
* **How it works:** The node applies a dense grid of points across the input image and generates a segmentation mask for the object located at each point. It features built-in multithreading to accelerate CPU inference and supports CUDA-enabled GPUs for heavy workloads. *Note: Users must download a SAM checkpoint file (e.g., `sam_vit_b_01ec64.pth`) locally to run this node.*
* **Key Parameters:**
    * **Model Configuration:** Requires the local path to the `.pth` weights file and the corresponding model type (`vit_b`, `vit_l`, or `vit_h`).
    * **Hardware Acceleration:** Toggle between CPU and GPU. For CPU users, the `CPU Threads` parameter allows allocating multiple cores to drastically reduce processing time.
    * **Advanced Tuning:** 
	    * *Points per side:* Defines the density of the sampling grid (default 16). Higher values find smaller objects but increase computation time.
        * *Number of crop layers:* Set to 0 for standard images, or increase to recursively crop and segment high-resolution images for extreme detail.
        * *Thresholds:* Fine-tune mask filtering using `Predicted IoU`, `Stability score`, and `Minimum mask region area` to discard low-quality or noisy masks.
* **Input:** A KNIME table containing a column of images and an ID column (Path).
 * **Output 1 (Segmented Image):** A table containing the original images overlaid with colorful, semi-transparent masks for every detected object, allowing for instant visual verification.
 * **Output 2 (Segmentation Results):** A detailed dataset containing the bounding box coordinates (`X_center`, `Y_center`, `Width`, `Height`), `Predicted IoU`, and `Stability Score` for every individual mask generated.
* **Use Cases:** Isolating biological structures (e.g., cells or tissues) in medical imaging, automatically extracting product silhouettes for e-commerce catalogs, or extracting complex geometric shapes and areas for downstream spatial analysis.

## Support
For any request about this software, please refer to the authors.

## Roadmap
This is a preliminary version. Further improvements and nodes will come soon. Stay tuned.

## Contributing
We welcome contributions to this project! To contribute:

- Fork this repository.
- Create a new branch.
- Make your changes and commit them.
- Submit a pull request.

Please make sure your code follows the existing style and includes appropriate tests.

## Authors and acknowledgment
Thanks to the collaboration with KNIME.

## License
The nodes are released under GPLv3.

## Project status
We are currently developing new nodes and improving the existing ones. Please feel free to contribute.
