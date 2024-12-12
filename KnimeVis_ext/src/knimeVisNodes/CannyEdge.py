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
    icon_path="icons/image_edge.png",
    category=knimeVis_category,
    id="canny-image"
)
@knext.input_table(name="Input Table", description="Table containing the image file path.")
@knext.output_table(name="Processed Image Table", description="Table containing processed images.")
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
    append_column = knext.ColumnParameter(
        label="Append Column",
        description="Select a column to append to the output, Leave empty for only the Canny edge detection column.",
        include_none_column=True,
        port_index=0,
    )
        # Parameters specific to Canny Edge Detection
    low_threshold = knext.IntParameter(
        label="Canny Low Threshold",
        description="Low threshold for Canny edge detection.",
        default_value=50,
        min_value=50,
        max_value=100,
    )
    high_threshold = knext.IntParameter(
        label="Canny High Threshold",
        description="High threshold for Canny edge detection.",
        default_value=150,
        min_value=150,
        max_value=300,
    )

    def cannyfunc(self, image, thresholds):
        """
        Applies Canny Edge Detection to the given image bu using OpenCV.
        """
        low_threshold, high_threshold = thresholds
        edges = cv.Canny(image, low_threshold, high_threshold)
        return Image.fromarray(edges)

    def configure(self, configure_context: knext.ConfigurationContext, input_schema_1: knext.Schema):
        # Validate and select the image column only
        image_columns = [c for c in input_schema_1 if kutil.is_png(c)]
        if not image_columns:
            raise ValueError("No valid image columns available for selection.")

        if self.image_column is None or self.image_column == "<none>":
            self.image_column = image_columns[-1].name  

        # Log selected image column
        LOGGER.info(f"Selected image column: {self.image_column}")

        # Ensure the image column exists in the input schema
        image_col = next((c for c in input_schema_1 if c.name == self.image_column), None)
        if image_col is None:
            raise ValueError(f"The column '{self.image_column}' does not exist in the input schema.")

        # by default , Define output schema with just the Canny edge detection column 
        output_columns = [
            knext.Column(knext.logical(Image.Image), "Canny Edge Detection"),  
        ]

        # If the user selected a column to append, add it to the output schema
        if self.append_column and self.append_column != "<none>":
            append_col = next((c for c in input_schema_1 if c.name == self.append_column), None)
            if append_col is None:
                raise ValueError(f"The column '{self.append_column}' does not exist in the input schema.")
            output_columns.append(append_col)  

        return knext.Schema.from_columns(output_columns)

    def execute(self, exec_context, input_table):
        # Convert KNIME table to DataFrame
        input_df = input_table.to_pandas()

        # Ensure the column contains valid PIL Images
        if self.image_column not in input_df.columns:
            raise ValueError(f"Column '{self.image_column}' not found in the input data frame.")

        # Check if each value in the column is a PIL Image
        if not all(isinstance(img, Image.Image) for img in input_df[self.image_column]):
            raise ValueError(f"Column '{self.image_column}' does not contain valid PIL.Image objects.")

        # Define a function to process each image
        def process_image(image_input):
            try:
                # Since the input is a PIL.Image object, we can directly process it
                if not isinstance(image_input, Image.Image):
                    raise ValueError(f"Expected a PIL.Image object, got {type(image_input)}.")

                # Convert the image to grayscale
                gray_image = np.array(image_input.convert("L"))

                # Apply Canny edge detection
                edges = self.cannyfunc(gray_image, (self.low_threshold, self.high_threshold))
                return edges

            except Exception as e:
                raise ValueError(f"Error processing image: {e}")

        # Apply the function and generate the Canny edge detection results
        input_df["Canny Edge Detection"] = input_df[self.image_column].apply(process_image)

        # Prepare output DataFrame
        output_columns = ["Canny Edge Detection"]

        # If user has selected an append column, include it in the output
        if self.append_column and self.append_column != "<none>":
            output_columns.append(self.append_column)

        # Generate the output table
        output_df = input_df[output_columns]

        return knext.Table.from_pandas(output_df)

    def cannyfunc(self, image, thresholds):
        low_threshold, high_threshold = thresholds
        edges = cv.Canny(image, low_threshold, high_threshold)
        return Image.fromarray(edges)






    