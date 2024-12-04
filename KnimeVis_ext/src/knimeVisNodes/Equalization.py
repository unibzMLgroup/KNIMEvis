import json
import cv2
import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
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

# Define the KNIME node
@knext.node(
    name="Histogram Equalization Node",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icon.png",
    category=knimeVis_category,
    id="eq-image"
)
@knext.input_table(name="Input Image Path", description="Table containing the image file path.")
@knext.output_table(name="Output Table", description="Table with Histogram details.")
class HistogramEqualizationNode:
    image_path_column = knext.ColumnParameter(
        "Image Location Column",
        "Select the column containing the image file path",
        include_none_column=False,
        column_filter=kutil.is_string
    )

    def configure(self, config_context, input_schema):
        # Validate that the selected column exists in the input schema
        if self.image_path_column not in input_schema.column_names:
            raise ValueError(f"Column '{self.image_path_column}' not found in input table.")
        return None

    def execute(self, exec_context, input_table):
        # Convert KNIME ArrowTable or dictionary to pandas DataFrame
        if hasattr(input_table, "to_pandas"):
            input_df = input_table.to_pandas()
        elif isinstance(input_table, pd.DataFrame):
            input_df = input_table
        elif isinstance(input_table, dict):
            input_df = pd.DataFrame.from_dict(input_table)
        else:
            raise TypeError(f"Unexpected input_table type: {type(input_table)}")

        # Validate the selected column
        if self.image_path_column not in input_df.columns:
            raise ValueError(f"Column '{self.image_path_column}' not found in input table.")

        # Initialize output lists
        Original_hist = []
        hist_equalize = []
        transfer_functions = []
        
        # Histogram Equalization Functions
        def imhist(im):
            m, n = im.shape
            h = [0.0] * 256
            for i in range(m):
                for j in range(n):
                    h[im[i, j]] += 1
            return np.array(h) / (m * n)

        def cumsum(h):
            return [sum(h[:i + 1]) for i in range(len(h))]

        def histeq(im):
            h = imhist(im)
            cdf = np.array(cumsum(h))
            sk = np.uint8(255 * cdf)
            s1, s2 = im.shape
            Y = np.zeros_like(im)
            for i in range(s1):
                for j in range(s2):
                    Y[i, j] = sk[im[i, j]]
            H = imhist(Y)
            return Y, h, H, sk

        # Process each image
        for image_path in input_df[self.image_path_column]:
             
            """
            try:
                path_obj = Path(str(image_path))  # Force conversion to Path object
                image_path_str = path_obj.as_posix()  # Convert Path object to string
                print(f"Processed path: {image_path_str}")
            except Exception as e:
                print(f"Failed to process path: {e}")
            """
            # Load the image
            image = np.uint8(Image.open(image_path))  # Ensure 8-bit grayscale
            if image.ndim == 3:  # Convert to grayscale if RGB
                image = np.uint8(0.2126 * image[:, :, 0] +
                                 0.7152 * image[:, :, 1] +
                                 0.0722 * image[:, :, 2])
            
            # Apply histogram equalization
            equalized_image, h, new_h, sk = histeq(image)

            # Save results
            Original_hist.append(h.tolist())  
            hist_equalize.append(new_h.tolist())
            transfer_functions.append(sk.tolist())
        
        # Create the output DataFrame
        result_df = pd.DataFrame({
            "Original Histogram": [json.dumps(img) for img in Original_hist],
            "Histogram Equalized": [json.dumps(hist) for hist in hist_equalize],
            "Transfer Function": [json.dumps(tf) for tf in transfer_functions],
        })

        # Return the DataFrame as a KNIME table
        return knext.Table.from_pandas(result_df)

