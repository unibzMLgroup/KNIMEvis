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

class SpatialFiltering:
    """
    Spatial Filtering Node

    The Spatial Filtering Node is a custom KNIME node designed for performing spatial filtering on images.
    """

    # define your parameter
    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select the column to apply spatial filtering.",
        port_index=0,
        column_filter=kutil.is_png
    )

    class AlgorithmOptions(knext.EnumParameterOptions):
        SOBEL = ("Sobel", "Description..")
        LAPLACE = ("Laplace", "Description..")
        ROBERT = ("Robert", "Description..")

    # TODO add parameter depending on the selected algorithm
    algorithm_selection_param = knext.EnumParameter(
        label="Spatial Filtering Algorithm Selection",
        description="Select the algorithm to produce spatial filtering",
        default_value=AlgorithmOptions.SOBEL.name,
        enum=AlgorithmOptions)
    
    
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

        # Log available image columns
        LOGGER.info(f"Available image columns: {image_columns[-1]}")

        # Set default column if not already set
        if self.image_column is None:
            self.image_column = image_columns[-1][0]
       
        # Log selected column
        LOGGER.warning(f"Selected image column: {self.image_column}")

        # Return the updated schema
        output_schema = input_schema_1.append(
            [knext.Column(knext.logical(Image.Image), "Filtering")])

        return output_schema
    
   
    def execute(self, execute_context: knext.ExecutionContext, input_table: knext.Table) -> knext.Table:

        df = input_table.to_pandas()
        images = df[self.image_column]
        # selected_algorithm = execute_context.node.get_value("algorithm_selection_param")

        if self.algorithm_selection_param == self.AlgorithmOptions.ROBERT.name:
            # Execute logic for Algorithm1
            df["Filtering"] = [self.robert(i)for i in images ]
        elif self.algorithm_selection_param == self.AlgorithmOptions.SOBEL.name:
            # Execute logic for Algorithm2
            df["Filtering"] = [self.sobel(i) for i in images]
        elif self.algorithm_selection_param == self.AlgorithmOptions.LAPLACE.name:
            # Execute logic for Algorithm3
            df["Filtering"] = [self.laplace(i) for i in images]
        else:
            raise ValueError(f"Unexpected algorithm: {self.algorithm_selection_param}")
            
        return knext.Table.from_pandas(df)
    
    # HACK to work with PIL
    def robert(self,img):
        img = np.array(img.convert("L"), dtype=np.float32)
        r, c = img.shape
        new_img = np.zeros((r,c)) 
        r_sunnzi = [[-1,-1],[1,1]]
        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild = img[x:x+2, y:y+2]
                    list_robert = r_sunnzi*imgChild
                    new_img[x, y] = abs(list_robert.sum()) # sum and absolute value
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)
        return Image.fromarray(new_img)
                    
    # # The implementation of sobel operator
    def sobel(self,img):
        img = np.array(img.convert("L"), dtype=np.float32)
        r, c = img.shape
        new_image = np.zeros((r, c))
        new_imageX = np.zeros(img.shape)
        new_imageY = np.zeros(img.shape)
        s_suanziX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # X direction
        s_suanziY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])     
        for i in range(r-2):
            for j in range(c-2):
                new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
                new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
                new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5

        new_img = np.clip(new_img, 0, 255).astype(np.uint8)
        return Image.fromarray(new_img)
    
    # Laplace operator
    def laplace(self,img):
        img = np.array(img.convert("L"), dtype=np.float32)
        r, c = img.shape
        new_image = np.zeros((r, c))
        L_sunnzi = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])          
        for i in range(r-2):
            for j in range(c-2):
                new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)
        return Image.fromarray(new_img)
    