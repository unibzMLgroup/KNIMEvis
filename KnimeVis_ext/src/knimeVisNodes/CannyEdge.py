import json, logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
import matplotlib.pyplot as plt
from typing import List
import cv2 as cv
from utils import knutills as kutil

LOGGER = logging.getLogger(__name__)

# define sub-category here
knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)


# Define the New node to detect edges of the images 
@knext.node(
    name="Canny Edge Detection",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icon.png",
    category=knimeVis_category,
    id="canny-image"
)
@knext.input_table(name="Input Table", description="Table containing the image file path.")
@knext.output_table(name="Processed Image Table", description="Table containing paths to processed images.")
class ImageEdgeDetection:
    """
    Canny Edge Detection Node for Image Processing Using OpenCV
    
    The Canny Edge Detection Node leverages OpenCV to perform efficient and accurate edge detection on images, seamlessly integrating into KNIME workflows for advanced image processing. This node allows users to select an input column containing images and apply the Canny Edge Detection algorithm, which identifies object boundaries and features by highlighting edges. 
    
    Images are automatically converted to grayscale before processing, and users can customize the low and high thresholds to fine-tune detection results. The processed images with detected edges are output as a new column in the KNIME table, making this node ideal for tasks such as feature extraction, object recognition, and preprocessing in computer vision.
    
    """
    image_column = knext.ColumnParameter(
        label= "Image Column",
        description= "Select the column containing the image.",
        include_none_column=False,
        column_filter =kutil.is_png
    )

    # Parameters specific to Canny Edge Detection
    low_threshold = knext.IntParameter(
        label="Canny Low Threshold",
        description= "Low threshold for Canny edge detection.",
        default_value=50
        
    )
    high_threshold = knext.IntParameter(
        label="Canny High Threshold",
        description="High threshold for Canny edge detection.",
        default_value=150
    )

    def cannyfunc(self, image, thresholds):
        """
        Applies Canny Edge Detection to the given image.
        """
        low_threshold, high_threshold = thresholds
        edges = cv.Canny(image, low_threshold, high_threshold)
        return Image.fromarray(edges)

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c) ]

        if not image_columns:
            # If no string columns are found, raise an exception or return None
            raise ValueError("No string columns available for image paths.")

        # Log available image columns
        LOGGER.info(f"Available image path columns: {image_columns[-1]}")

        # Set default column if not already set
        if self.image_column is None:
            self.image_column = image_columns[-1][0]
       
        # Log selected column
        LOGGER.warning(f"Selected image path column: {self.image_column}")
        
        # Return the updated schema
        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "Canny Edge Detection")])

        return output_schema
    
    def execute(self, exec_context, input_table):
        # Convert KNIME ArrowTable or dictionary to pandas DataFrame if needed
        if hasattr(input_table, "to_pandas"):
            input_df = input_table.to_pandas()
        elif isinstance(input_table, pd.DataFrame):
            input_df = input_table
        elif isinstance(input_table, dict):
            input_df = pd.DataFrame.from_dict(input_table)
        else:
            raise TypeError(f"Unexpected input_table type: {type(input_table)}")

        # Define a function to process each image
        def Image_edges(image):
            # Convert the image to grayscale if necessary
            if isinstance(image, np.ndarray):  # OpenCV format
                gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            elif isinstance(image, Image.Image):  # PIL format
                gray_image = np.array(image.convert("L"))
            else:
                raise TypeError("Unsupported image format. Expected PIL Image or OpenCV matrix.")

            edges = self.cannyfunc(gray_image, (self.low_threshold, self.high_threshold))
            # Convert the edge-detected image back to a PIL image
            #edge_image = Image.fromarray(edges.astype(np.uint8))

            return edges

        # Apply the process_image function to all images in the column
        input_df["Canny Edge Detection"] = input_df[self.image_column].apply(Image_edges)

        # Return the updated DataFrame as a KNIME table
        return knext.Table.from_pandas(input_df)
