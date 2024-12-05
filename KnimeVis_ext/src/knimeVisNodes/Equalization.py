import json
import logging
import numpy as np
import pandas as pd
from PIL import Image
import knime.extension as knext
import matplotlib.pyplot as plt
from typing import List
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

# Define the KNIME node for Equalization
@knext.node(
    name="Equalization",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icon.png",
    category=knimeVis_category,
    id="eq-image"
)
@knext.input_table(
    name="Input Image Path", 
    description="Table containing the image file path."
    )
@knext.output_table(
    name="Output Table", 
    description="Table with Histogram details."
    )

class Equalization:
    """    
    Image Histogram Equalization and Analysis

    The Image Histogram Equalization and Analysis node enhances the contrast of grayscale or RGB images through histogram equalization, making it an essential tool for image preprocessing workflows. It accepts a table containing image file paths, processes the images, and outputs a table enriched with detailed insights into the histogram equalization process. This node is particularly useful for computer vision applications, educational demonstrations, and preprocessing steps in machine learning pipelines.

    The output table includes the Equalized Image, showcasing the improved contrast after processing, and three additional columns providing detailed analysis: the Original Histogram of the input image, the Histogram Equalized values post-processing, and the Transfer Function, which maps intensity levels from the original image to the equalized result. These outputs enable users to understand the transformation and its impact on image intensity distribution.

    """
    image_column = knext.ColumnParameter(
        "Select Image Path ",
        "Select the column containing the image file path",
        include_none_column=False,
        port_index=0,
        column_filter=kutil.is_png
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c) ]

        if not image_columns:
            raise ValueError("No string columns available for image paths.")

        LOGGER.info(f"Available image path columns: {image_columns[-1]}")

        # Set default column if not already set
        if self.image_column is None:
            self.image_column = image_columns[-1][0]
       
        # Log selected column
        LOGGER.warning(f"Selected image path column: {self.image_column}")
        
        # updated schema
        output_schema = input_schema_1.append([
        knext.Column(knext.logical(Image.Image), "Equalized Image"),
        knext.Column(knext.string(), "Original Histogram"),
        knext.Column(knext.string(), "Histogram Equalized"),
        knext.Column(knext.string(), "Transfer Function"),
    ])

        return output_schema

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
        
        # A function to process and compute the matrix
        def process_images(image_paths):
            
            for image_path in image_paths:
                # Load the image
                image = np.array(image_path.convert("L"))
                #image = np.uint8(Image.open(image_path))  # Ensure 8-bit grayscale
                if image.ndim == 3:  # Convert to grayscale if RGB
                    image = np.uint8(0.2126 * image[:, :, 0] +
                                    0.7152 * image[:, :, 1] +
                                    0.0722 * image[:, :, 2])

                # Apply histogram equalization
                equalized_image, h, new_h, sk = histeq(image)

                # Save results to lists for JSON columns
                Original_hist.append(h.tolist())
                hist_equalize.append(new_h.tolist())
                transfer_functions.append(sk.tolist())

                # Yield the equalized image and also include the histograms in the table
                yield equalized_image, h.tolist(), new_h.tolist(), sk.tolist()

        # Apply the process_images function to all rows in the image column
        equalized_images = []
        original_histograms = []
        hist_equalized = []
        transfer_functions_list = []
        
        for image, hist, eq_hist, tf in process_images(input_df[self.image_column]):
            equalized_images.append(Image.fromarray(image))  
            original_histograms.append(json.dumps(hist))
            hist_equalized.append(json.dumps(eq_hist))
            transfer_functions_list.append(json.dumps(tf))

        # Add the equalized images and histogram data to the DataFrame
        input_df["Equalized Image"] = equalized_images
        input_df["Original Histogram"] = original_histograms
        input_df["Histogram Equalized"] = hist_equalized
        input_df["Transfer Function"] = transfer_functions_list

        # Return the DataFrame as a KNIME table
        return knext.Table.from_pandas(input_df)