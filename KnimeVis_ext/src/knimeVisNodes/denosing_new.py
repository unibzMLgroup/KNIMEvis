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
    name="Denoising New V2",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/denoise.png",
    category=knimeVis_category,
    id="denoise-v2-image",
)
@knext.output_table(
    name="Image Data",
    description="Table containing denoised images",
)
@knext.input_table(
    name="Image Data",
    description="Table containing image",
)
class DenoisingNewV2:
    """
    Denoising New Node
    
    A custom KNIME node that applies denoising to images using selected algorithms.
    Currently, it supports Median Filtering and Gaussian Filtering.
    """

    # Input column selection
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply Denoising.",
        port_index=0,
        column_filter=kutil.is_png  # Filter for image columns
    )

    class AlgorithmOptions(knext.EnumParameterOptions):
        MEDIAN = ("Median Filtering", "Applies a median filter to reduce noise.")
        GAUSSIAN = ("Gaussian Filtering", "Applies a Gaussian filter to reduce noise.")

    algorithm_selection_param = knext.EnumParameter(
        label="Denoising Algorithm Selection",
        description="Select the algorithm to produce denoised images.",
        default_value=AlgorithmOptions.MEDIAN.name,
        enum=AlgorithmOptions
    )

    filter_size = knext.IntParameter(
        label="Filter size",
        description="Value between 3 to 15 to specify the size of the filter box.",
        default_value=3,
        min_value=3,
        max_value=15,
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

        if self.filter_size % 2 == 0:
            raise ValueError("Filter size must be an odd integer.")

        # Return the updated schema
        output_schema = input_schema_1.append(
            [
                knext.Column(knext.logical(Image.Image), "Denoised Image"),  # Denoised image column
                knext.Column(knext.string(), "Algorithm Used"),             # Algorithm used column
            ]
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
                    if self.algorithm_selection_param == "MEDIAN":
                        return self.median_filter(image, self.filter_size)
                    elif self.algorithm_selection_param == "GAUSSIAN":
                        return self.gaussian_filter(image, self.filter_size)
                else:
                    image = Image.open(image)
                    if self.algorithm_selection_param == "MEDIAN":
                        return self.median_filter(image, self.filter_size)
                    elif self.algorithm_selection_param == "GAUSSIAN":
                        return self.gaussian_filter(image, self.filter_size)
            except Exception as e:
                LOGGER.error(f"Error processing image: {e}")
                return None

        # Parallel processing of images
        with ThreadPoolExecutor(max_workers=16) as executor:
            df["Denoised Image"] = list(executor.map(process_image, images))

        # Add the selected algorithm to the table
        df["Algorithm Used"] = self.algorithm_selection_param

        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        LOGGER.info(f"Node execution completed in {elapsed_time:.2f} seconds.")

        # Return updated KNIME table
        return knext.Table.from_pandas(df)

    def median_filter(self, image, filter_size):
        # Convert image to NumPy array
        data = np.array(image.convert("L"), dtype=np.uint8)
        # Apply OpenCV median filter
        denoised = cv.medianBlur(data, filter_size)
        # Convert back to PIL image
        return Image.fromarray(denoised)

    def gaussian_filter(self, image, filter_size):
        # Convert image to NumPy array
        data = np.array(image.convert("L"), dtype=np.uint8)
        # Apply Gaussian blur filter
        denoised = cv.GaussianBlur(data, (filter_size, filter_size), 0)
        # Convert back to PIL image
        return Image.fromarray(denoised)
