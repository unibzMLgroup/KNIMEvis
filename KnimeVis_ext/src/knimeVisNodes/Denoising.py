import knime.extension as knext
import numpy as np
import logging
import cv2 as cv
from utils import knutills as kutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image 
import time
import os

# Logger setup
LOGGER = logging.getLogger(__name__)

# define sub-category here
knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)

# Node definition
@knext.node(
    name="Denoising",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/denoise.png",
    category=knimeVis_category,
    id="denoise-image",
)
@knext.output_table(
    name="Image Data",
    description="Table containing image",
)

@knext.input_table(
    name="Image Data",
    description="Table containing image",
)

class Denoising:
    """
    Denoising

    This node reduces noise in images by applying advanced filtering techniques, enhancing image clarity and quality for downstream analysis.
    """

    # define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply Denoising.",
        port_index=0,
        column_filter=kutil.is_png
    )

    class AlgorithmOptions(knext.EnumParameterOptions):
        MEDIAN = ("Median Filtering", "Removes noise by replacing each pixel's value with the median value of the surrounding pixels, preserving edges effectively.")
        MEDIAN_openCV = ("Median Filtering", "Removes noise by replacing each pixel's value with the median value of the surrounding pixels, preserving edges effectively.")
        GAUSSIAN_openCV = ("Gaussian Filtering", "Applies a Gaussian filter to reduce noise.")

    algorithm_selection_param = knext.EnumParameter(
        label="Denoising Algorithm Selection",
        description="Select the algorithm to produce denoised images.",
        default_value=AlgorithmOptions.MEDIAN.name,
        enum=AlgorithmOptions
    )

    algorithm_selection_param = knext.EnumParameter(
        label="Denoising Algorithm Selection",
        description="Choose the algorithm to apply for reducing noise and producing clearer, denoised images.",
        default_value=AlgorithmOptions.MEDIAN.name,
        enum=AlgorithmOptions)
    
    filter_size =  knext.IntParameter(
        label="Filter size",
        description="Specify an odd value to define the size of the filter box used for image denoising.",
        default_value=3,
        min_value=3,
        max_value=15,
    )
    
    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):
        
        # Filter string columns for image 
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c) ]

        if not image_columns:
            # If no string columns are found, raise an exception or return None
            raise ValueError("No string columns available for image paths.")
        
        # Set default column if not already set
        if self.image_column is None:
            self.image_column = image_columns[-1][0]
       
        # Log selected column
        LOGGER.info(f"Selected image column: {self.image_column}")

        if self.filter_size % 2 == 0:
            raise ValueError("Filter size must be an odd integer.")
        
        # Return the updated schema
        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "Denoising")])

        return output_schema
    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:

        # Start timing
        start_time = time.time()
        
        df = input_table.to_pandas()
        images = df[self.image_column]
        # selected_algorithm = execute_context.node.get_value("algorithm_selection_param")

        # Helper function to process each image
        def process_image(image):
            try:
                if self.algorithm_selection_param == self.AlgorithmOptions.MEDIAN.name:
                    # Execute logic for Algorithm1
                    df["Denoising"] = [self.median_filter(i,self.filter_size)for i in images ]

                elif self.algorithm_selection_param == self.AlgorithmOptions.MEDIAN_openCV.name:
                    # Execute logic for Algorithm1
                    df["Denoising"] = [self.median_filter_opencv(i,self.filter_size)for i in images ]
                elif self.algorithm_selection_param == self.AlgorithmOptions.GAUSSIAN_openCV.name:
                    # Execute logic for Algorithm1
                    df["Denoising"] = [self.gaussian_filter_opencv(i,self.filter_size)for i in images ]
                else:
                    raise ValueError(f"Unexpected algorithm: {self.algorithm_selection_param}")
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

        
            
        return knext.Table.from_pandas(df)
    
    # HACK to work with PIL
    def median_filter(self,data, filter_size):
        temp = []
        data = np.array(data.convert("L"),dtype=np.uint8)
        indexer = filter_size // 2
        data_final = []
        data_final = np.zeros((len(data),len(data[0])))
        for i in range(len(data)):

            for j in range(len(data[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(data[i + z - indexer][j + k - indexer])
                
                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
        # HACK to produce images, correct if the case
        new_img = np.clip(data_final, 0, 255).astype(np.uint8)
        return Image.fromarray(new_img)
    
    def median_filter_opencv(self, image, filter_size):
        # HACK to work with PIL
        # Convert image to NumPy array
        data = np.array(image.convert("L"), dtype=np.uint8)
        # Apply OpenCV median filter
        denoised = cv.medianBlur(data, filter_size)
        # Convert back to PIL image
        return Image.fromarray(denoised)
    
    def gaussian_filter_opencv(self, image, filter_size):
        # Convert image to NumPy array
        data = np.array(image.convert("L"), dtype=np.uint8)
        # Apply Gaussian blur filter
        denoised = cv.GaussianBlur(data, (filter_size, filter_size), 0)
        # Convert back to PIL image
        return Image.fromarray(denoised)

    