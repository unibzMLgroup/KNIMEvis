import knime.extension as knext
import numpy as np
import logging
import cv2 as cv
from utils import knutills as kutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

# Logger setup
LOGGER = logging.getLogger(__name__)

# Define sub-category
knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)

# Node definition
@knext.node(
    name="Spatial Filtering - Sobel",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/denoise.png",
    category=knimeVis_category,
    id="sobel-edge-detection",
)
@knext.output_table(
    name="Edge Detected Image",
    description="Table containing edge detected images using Sobel filter",
)
@knext.input_table(
    name="Image Data",
    description="Table containing images for edge detection",
)
class SobelEdgeDetection:
    """
    Sobel Edge Detection Node
    
    A custom KNIME node that applies Sobel filtering for edge detection on images.
    """

    # Input column selection
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply Sobel edge detection.",
        port_index=0,
        column_filter=kutil.is_png
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        # Filter string columns for image paths
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]

        if not image_columns:
            raise ValueError("No string columns available for image paths.")

        # Set default column if not already set
        if self.image_column is None:
            self.image_column = image_columns[-1][0]

        # Return the updated schema
        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "Edge Detected Image")]  # اضافه کردن ستونی برای تصاویر لبه‌دار
        )

        return output_schema

    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:
        # Start timing
        start_time = time.time()

        # Convert KNIME table to Pandas DataFrame
        df = input_table.to_pandas()
        images = df[self.image_column]

        # Helper function to process each image
        def process_image(image):
            try:
                if isinstance(image, Image.Image):
                    return self.sobel_edge_detection(image)
                else:
                    image = Image.open(image)
                    return self.sobel_edge_detection(image)
            except Exception as e:
                LOGGER.error(f"Error processing image: {e}")
                return None

        # Parallel processing of images
        with ThreadPoolExecutor(max_workers=16) as executor:
            df["Edge Detected Image"] = list(executor.map(process_image, images))

        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        LOGGER.info(f"Node execution completed in {elapsed_time:.2f} seconds.")

        # Return updated KNIME table
        return knext.Table.from_pandas(df)

    def sobel_edge_detection(self, image):
        # Convert image to grayscale
        gray_image = np.array(image.convert("L"), dtype=np.uint8)

        # Apply Gaussian blur to reduce noise (optional but recommended)
        blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)

        # Apply Sobel filter in both X and Y directions
        sobel_x = cv.Sobel(blurred_image, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(blurred_image, cv.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude (sqrt(sobel_x^2 + sobel_y^2))
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 8-bit image
        gradient_magnitude = np.clip(gradient_magnitude / np.max(gradient_magnitude) * 255, 0, 255).astype(np.uint8)

        # Apply thresholding to keep only significant edges (adjust threshold as needed)
        _, thresholded_image = cv.threshold(gradient_magnitude, 50, 255, cv.THRESH_BINARY)

        # Convert back to an image
        return Image.fromarray(thresholded_image)
