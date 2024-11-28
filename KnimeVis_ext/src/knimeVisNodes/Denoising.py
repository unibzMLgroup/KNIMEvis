import knime.extension as knext
import numpy as np
import logging
import cv2 as cv
from utils import knutills as kutil
from PIL import Image 
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
    name="Spatial Filtering",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/filterIcon.png",
    category=knimeVis_category,
    id="filtering-image",
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

    The Denoising Node is a custom KNIME node designed for performing denoising on images.
    """

    # define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply Denoising.",
        port_index=0,
        column_filter=kutil.is_png
    )

    class AlgorithmOptions(knext.EnumParameterOptions):
        MEDIAN = ("Median Filtering", "Description..")


    algorithm_selection_param = knext.EnumParameter(
        label="Denoising Algorithm Selection",
        description="Select the algorithm to produce Denoised Immages",
        default_value=AlgorithmOptions.MEDIAN.name,
        enum=AlgorithmOptions)
    
    filter_size =  knext.IntParameter(
        label="Filter size",
        description="Value between 0 to 9 to toggle size of filter box used for filtering",
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

        df = input_table.to_pandas()
        images = df[self.image_column]
        # selected_algorithm = execute_context.node.get_value("algorithm_selection_param")

        if self.algorithm_selection_param == self.AlgorithmOptions.MEDIAN.name:
            # Execute logic for Algorithm1
            df["Denoising"] = [self.median_filter(i,self.filter_size)for i in images ]
        else:
            raise ValueError(f"Unexpected algorithm: {self.algorithm_selection_param}")
            
        return knext.Table.from_pandas(df)
    
    # HACK to work with PIL
    def median_filter(data, filter_size):
        temp = []
        data = np.array(data, dtype=np.float32)
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
        return Image.fromarray(data_final)
    