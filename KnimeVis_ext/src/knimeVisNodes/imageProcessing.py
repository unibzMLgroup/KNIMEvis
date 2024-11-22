import knime.extension as knext
from utils import knutills as kutil
import logging
from PIL import Image, ImageOps, ImageFilter
import io
import pandas as pd

LOGGER = logging.getLogger(__name__)


# define sub-category here
knimeVis_category = knext.category(
    path="/community/unibz/",
    level_id="imageProc",
    name="Image Processing",
    description="Python Nodes for Image Processing",
    icon="icons/knimeVisLab32.png",
)

# necessary decorators to define the structure of the node
@knext.node(
    name="Image Processor (Table)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icon.png",
    category=knimeVis_category,
    id="ímg-blurrer-table",
)
@knext.input_table(
    name="Input Data",
    description="Table containing the image column.",
)
@knext.output_table(
    name="Some output",
    description="Output table containing the image column.",
)
class ImageProcessorNode:
    """
    A first example node for UniBZ KnimeVis extension.

    This node can be used to apply some basic image processing algorithms on PNG images
    """

    # define your parameter
    target_column = knext.ColumnParameter(
        label="Image Column",
        description="Select an Image column for blurring.",
        port_index=0,
        column_filter=kutil.is_png,
    )

    radius_box_blur = knext.IntParameter(
        label="Radius",
        description="Value between 0 to 9 to toggle size of square box used for blurring in each direction.",
        default_value=2,
        min_value=0,
        max_value=10,
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema_1: knext.Schema,
    ):

        # pick only image columns
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]

        LOGGER.debug(f"ImageProcessorNode configure: {image_columns}" )

        # by default select the last column
        if self.target_column is None:
            self.target_column = image_columns[-1][0]

        # output_schema =
        return input_schema_1.append(
            [
                knext.Column(knext.list_(knext.int64()), "Image Dimensions"),
                knext.Column(knext.logical(Image.Image), "Flipped"),
                knext.Column(knext.logical(Image.Image), "Blurred"),
            ]
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        LOGGER.debug("ImageProcessorNode execute:...")
        
        df = input_table.to_pandas()
        images = df[self.target_column]
        df["Image Dimensions"] = [list(i.size) for i in images]
        df["Flipped"] = [ImageOps.flip(i) for i in images]
        df["Blurred"] = [
            i.filter(ImageFilter.BoxBlur(radius=self.radius_box_blur)) for i in images
        ]

        return knext.Table.from_pandas(df)
